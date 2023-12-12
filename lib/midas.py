import os
import glob
import torch
import cv2
import matplotlib.pyplot as plt
import os

import numpy as np
import torch.fft as fft
import ipdb
import copy
import wget

from midas.model_loader import load_model
import torch.nn.functional as F
first_execution = True
thisdir = os.path.abspath(os.path.dirname(__file__))

class MiDas():
    def __init__(self, device, model_type) -> None:
        self.device = device

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        model_weights = os.path.join(thisdir, '..' ,f"./weights/{model_type}.pt")
        if not os.path.exists(model_weights):
            os.makedirs(os.path.dirname(model_weights), exist_ok=True)
            if '384' in model_type:
                wget.download('https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt', model_weights)
            elif '512' in model_type:
                wget.download('https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt', model_weights)
            else:
                assert False, 'please select correct depth estimation model.'
        print("Device: %s" % device)
        model, transform, net_w, net_h = load_model(
            device, model_weights, model_type, optimize=False, height=None, square=False
        )
        self.model = model
        self.transform = transform
        self.model_type = model_type
        self.net_w = net_w
        self.net_h = net_h

    def process(
        self, device, model, model_type, image, input_size, target_size, optimize, use_camera
    ):
        """
        Run the inference and interpolate.

        Args:
            device (torch.device): the torch device used
            model: the model used for inference
            model_type: the type of the model
            image: the image fed into the neural network
            input_size: the size (width, height) of the neural network input (for OpenVINO)
            target_size: the size (width, height) the neural network output is interpolated to
            optimize: optimize the model to half-floats on CUDA?
            use_camera: is the camera used?

        Returns:
            the prediction
        """
        global first_execution

        if "openvino" in model_type:
            if first_execution or not use_camera:
                # print(
                #     f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder"
                # )
                first_execution = False

            sample = [np.reshape(image, (1, 3, *input_size))]
            prediction = model(sample)[model.output(0)][0]
            prediction = cv2.resize(
                prediction, dsize=target_size, interpolation=cv2.INTER_CUBIC
            )
        else:
            sample = torch.from_numpy(image).to(device).unsqueeze(0)

            if optimize and device == torch.device("cuda"):
                if first_execution:
                    print(
                        "  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                        "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                        "  half-floats."
                    )
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            if first_execution or not use_camera:
                height, width = sample.shape[2:]
                print(f"    Input resized to {width}x{height} before entering the encoder")
                first_execution = False

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=target_size[::-1],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        return prediction
    
    def prediction2depth(self, depth):
        bits = 1
        if not np.isfinite(depth).all():
            depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            print("WARNING: Non-finite depth values present")
        
        depth_min = depth.min()
        depth_max = depth.max()

        max_val = (2**(8*bits))-1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)
        # out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)
        return out
    
    def calc_R(self, theta_z, theta_x, theta_y):
        theta_z, theta_x, theta_y = theta_z/180*np.pi, theta_x/180*np.pi, theta_y/180*np.pi, 
        Rz = np.array([[np.cos(theta_z), np.sin(theta_z), 0],
                    [-np.sin(theta_z), np.cos(theta_z), 0],
                    [0,0,1]])
        Rx = np.array([[1,0,0],
                    [0,np.cos(theta_x), np.sin(theta_x)],
                    [0, -np.sin(theta_x), np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0,1,0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
        
        R = Rz @ Rx @ Ry
        return R 
    
    def render_new_view(self, img, depth, R, t, K):
        h, w, _ = img.shape
        new_img = np.zeros_like(img)

        for y in range(h):
            for x in range(w):
                # Back-project
                Z = depth[y, x]
                X = (x - K[0, 2]) * Z / K[0, 0]
                Y = (y - K[1, 2]) * Z / K[1, 1]
                point3D = np.array([X, Y, Z, 1])

                # Transform
                point3D_new = R @ point3D[:3] + t
                if point3D_new[2] <= 0:  # point is behind the camera
                    continue

                # Project to new view
                u = int(K[0, 0] * point3D_new[0] / point3D_new[2] + K[0, 2])
                v = int(K[1, 1] * point3D_new[1] / point3D_new[2] + K[1, 2])
                
                if 0 <= u < w and 0 <= v < h:
                    new_img[v, u] = img[y, x]
        return new_img
        
    def wrap_img(self, img, depth_map, K, R, T, target_point=None):
        h, w = img.shape[:2]
        # Generate grid of coordinates (x, y)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones_like(x)

        # Flatten and stack to get homogeneous coordinates
        homogeneous_coordinates = np.stack((x.flatten(), y.flatten(), ones.flatten()), axis=1).T

        # Inverse intrinsic matrix
        K_inv = np.linalg.inv(K)

        # Inverse rotation and translation
        R_inv = R.T
        T_inv = -R_inv @ T

        # Project to 3D using depth map
        world_coordinates = K_inv @ homogeneous_coordinates
        world_coordinates *= depth_map.flatten()

        # Apply inverse transformation
        transformed_world_coordinates = R_inv @ world_coordinates + T_inv.reshape(-1, 1)

        # Project back to 2D
        valid = transformed_world_coordinates[2, :] > 0
        projected_2D = K @ transformed_world_coordinates
        projected_2D /= projected_2D[2, :]

        # Initialize map_x and map_y
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        # Assign valid projection values to map_x and map_y
        map_x.flat[valid] = projected_2D[0, valid]
        map_y.flat[valid] = projected_2D[1, valid]

        # Perform the warping
        wrapped_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        if target_point is None:
            return wrapped_img
        else:
            target_point = (map_x[int(target_point[1]), int(target_point[0])], map_y[int(target_point[1]), int(target_point[0])])
            target_point = tuple(max(0, min(511, x)) for x in target_point)
            return wrapped_img, target_point

    def get_low_high_frequent_tensors(self, x, threshold=4):
        dtype = x.dtype
        x = x.type(torch.float32)

        # FFT
        x_freq = fft.fftn(x, dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))
        B,C,H,W = x_freq.shape
        mask = torch.ones((B, C, H, W)).to(x.device)

        crow, ccol = H // 2, W //2
        mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 0 # low 0 high 1
        x_freq_high = x_freq * mask
        x_freq_low = x_freq * (1 - mask)

        x_freq_high = fft.ifftshift(x_freq_high, dim=(-2, -1))
        x_high = fft.ifftn(x_freq_high, dim=(-2, -1)).real
        x_high = x_high.type(dtype)

        x_freq_low = fft.ifftshift(x_freq_low, dim=(-2, -1))
        x_low = fft.ifftn(x_freq_low, dim=(-2, -1)).real
        x_low = x_low.type(dtype)
        return x_high, x_low, x_freq_high, x_freq_low, mask
    
    def combine_low_and_high(self, freq_low, freq_high, mask):
        freq = freq_high * mask + freq_low * (1-mask)
        freq = fft.ifftshift(freq, dim=(-2, -1))
        x = fft.ifftn(freq, dim=(-2, -1)).real
        return x


    def wrap_img_tensor_w_fft(self, img_tensor, depth_tensor, 
                        theta_z=0, theta_x=0, theta_y=-10, T=[0,0,-2], threshold=4):
        _, img_tensor, high_freq, low_freq, fft_mask = self.get_low_high_frequent_tensors(img_tensor, threshold)

        intrinsic_matrix = np.array([[1000, 0, img_tensor.shape[-1]/2],
                    [0, 1000, img_tensor.shape[-2]/2],
                    [0, 0, 1]])  # Example intrinsic matrix
        ori_size = None
        if depth_tensor.shape[-1] != img_tensor.shape[-1]:
            scale = depth_tensor.shape[-1] / img_tensor.shape[-1]
            ori_size = (img_tensor.shape[-2], img_tensor.shape[-1])
            img_tensor_ori = img_tensor.clone()
            # img_tensor = F.interpolate(img_tensor, size=(depth_tensor.shape[-2], depth_tensor.shape[-1]))
            depth_tensor = F.interpolate(depth_tensor.unsqueeze(0).unsqueeze(0), size=ori_size, mode='bilinear').squeeze().to(torch.float16)
            intrinsic_matrix[0,0] /= scale
            intrinsic_matrix[1,1] /= scale
        rotation_matrix = self.calc_R(theta_z=theta_z, theta_x=theta_x, theta_y=theta_y)
        translation_vector = np.array(T)  # Translation vector to shift camera to the right

        h,w = img_tensor.shape[2:]

        xy_src = np.mgrid[0:h, 0:w].reshape(2, -1)

        xy_src_homogeneous = np.vstack((xy_src, np.ones((1, xy_src.shape[1]))))

        # Convert to torch tensors
        xy_src_homogeneous_tensor = torch.tensor(xy_src_homogeneous, dtype=torch.float16, device=img_tensor.device)

        # Compute the coordinates in the world frame
        xy_world = torch.inverse(torch.tensor(intrinsic_matrix, dtype=torch.float32, device=img_tensor.device)).to(torch.float16) @ xy_src_homogeneous_tensor
        xy_world = xy_world * depth_tensor.view(1, -1)

        # Compute the coordinates in the new camera frame
        xy_new_cam = torch.inverse(torch.tensor(rotation_matrix, dtype=torch.float32, device=img_tensor.device)).to(torch.float16) @ (xy_world - torch.tensor(translation_vector, dtype=torch.float16, device=img_tensor.device).view(3,1))

        # Compute the coordinates in the new image
        xy_dst_homogeneous = torch.tensor(intrinsic_matrix, dtype=torch.float16, device=img_tensor.device) @ xy_new_cam
        xy_dst = xy_dst_homogeneous[:2, :] / xy_dst_homogeneous[2, :]

        # Reshape to a 2D grid and normalize to [-1, 1]
        xy_dst = xy_dst.reshape(2, h, w)
        xy_dst = (xy_dst - torch.tensor([[w/2.0], [h/2.0]], dtype=torch.float16, device=img_tensor.device).unsqueeze(-1)) / torch.tensor([[w/2.0], [h/2.0]], dtype=torch.float16, device=img_tensor.device).unsqueeze(-1)
        xy_dst = torch.flip(xy_dst, [0])
        xy_dst = xy_dst.permute(1, 2, 0)

        # Perform the warping
        wrapped_img = F.grid_sample(img_tensor, xy_dst.to(torch.float16)[None], align_corners=True, mode='bilinear', padding_mode='reflection')
        wrapped_freq = fft.fftn(wrapped_img, dim=(-2, -1))
        wrapped_freq = fft.fftshift(wrapped_freq, dim=(-2, -1))
        wrapped_img = self.combine_low_and_high(wrapped_freq, high_freq, fft_mask)
        return wrapped_img
    
    def wrap_img_tensor_w_fft_ext(self, img_tensor, depth_tensor, K,R,T, threshold=4):
        _, img_tensor, high_freq, low_freq, fft_mask = self.get_low_high_frequent_tensors(img_tensor, threshold)

        ori_size = None

        if depth_tensor.shape[-1] != img_tensor.shape[-1]:
            scale = depth_tensor.shape[-1] / img_tensor.shape[-1]
            ori_size = (img_tensor.shape[-2], img_tensor.shape[-1])
            # img_tensor = F.interpolate(img_tensor, size=(depth_tensor.shape[-2], depth_tensor.shape[-1]))
            depth_tensor = F.interpolate(depth_tensor.unsqueeze(0).unsqueeze(0), size=ori_size, mode='bilinear').squeeze().to(torch.float16)
            intrinsic = copy.deepcopy(K)
            intrinsic = K / scale
            intrinsic[2,2] = 1

        h,w = img_tensor.shape[2:]

        xy_src = np.mgrid[0:h, 0:w].reshape(2, -1)

        xy_src_homogeneous = np.vstack((xy_src, np.ones((1, xy_src.shape[1]))))

        # Convert to torch tensors
        xy_src_homogeneous_tensor = torch.tensor(xy_src_homogeneous, dtype=img_tensor.dtype, device=img_tensor.device)

        # Compute the coordinates in the world frame
        # xy_world = torch.inverse(torch.tensor(K, dtype=torch.float32, device=img_tensor.device)).to(torch.float16) @ xy_src_homogeneous_tensor
        xy_world = torch.tensor(np.linalg.inv(intrinsic)).to(img_tensor.dtype).to(img_tensor.device) @ xy_src_homogeneous_tensor
        xy_world = xy_world * depth_tensor.view(1, -1)

        # Compute the coordinates in the new camera frame
        xy_new_cam = torch.inverse(torch.tensor(R, dtype=torch.float32, device=img_tensor.device)).to(img_tensor.dtype) @ (xy_world - torch.tensor(T, dtype=img_tensor.dtype, device=img_tensor.device).view(3,1))

        # Compute the coordinates in the new image
        xy_dst_homogeneous = torch.tensor(intrinsic, dtype=img_tensor.dtype, device=img_tensor.device) @ xy_new_cam
        xy_dst = xy_dst_homogeneous[:2, :] / xy_dst_homogeneous[2, :]

        # Reshape to a 2D grid and normalize to [-1, 1]
        xy_dst = xy_dst.reshape(2, h, w)
        xy_dst = (xy_dst - torch.tensor([[w/2.0], [h/2.0]], dtype=img_tensor.dtype, device=img_tensor.device).unsqueeze(-1)) / torch.tensor([[w/2.0], [h/2.0]], dtype=img_tensor.dtype, device=img_tensor.device).unsqueeze(-1)
        xy_dst = torch.flip(xy_dst, [0])
        xy_dst = xy_dst.permute(1, 2, 0)

        # Perform the warping
        wrapped_img = F.grid_sample(img_tensor, xy_dst.to(img_tensor.dtype)[None], align_corners=True, mode='bilinear', padding_mode='reflection')
        wrapped_freq = fft.fftn(wrapped_img, dim=(-2, -1))
        wrapped_freq = fft.fftshift(wrapped_freq, dim=(-2, -1))
        wrapped_img = self.combine_low_and_high(wrapped_freq, high_freq, fft_mask)
        return wrapped_img

    def wrap_img_tensor_w_fft_matrix(self, img_tensor, depth_tensor, 
                        theta_z=0, theta_x=0, theta_y=-10, T=[0,0,-2], threshold=4):
        _, img_tensor, high_freq, low_freq, fft_mask = self.get_low_high_frequent_tensors(img_tensor, threshold)

        intrinsic_matrix = np.array([[1000, 0, img_tensor.shape[-1]/2],
                    [0, 1000, img_tensor.shape[-2]/2],
                    [0, 0, 1]])  # Example intrinsic matrix
        ori_size = None
        if depth_tensor.shape[-1] != img_tensor.shape[-1]:
            scale = depth_tensor.shape[-1] / img_tensor.shape[-1]
            ori_size = (img_tensor.shape[-2], img_tensor.shape[-1])
            img_tensor_ori = img_tensor.clone()
            # img_tensor = F.interpolate(img_tensor, size=(depth_tensor.shape[-2], depth_tensor.shape[-1]))
            depth_tensor = F.interpolate(depth_tensor.unsqueeze(0).unsqueeze(0), size=ori_size, mode='bilinear').squeeze().to(torch.float16)
            intrinsic_matrix[0,0] /= scale
            intrinsic_matrix[1,1] /= scale
        rotation_matrix = self.calc_R(theta_z=theta_z, theta_x=theta_x, theta_y=theta_y)
        translation_vector = np.array(T)  # Translation vector to shift camera to the right

        h,w = img_tensor.shape[2:]

        xy_src = np.mgrid[0:h, 0:w].reshape(2, -1)

        xy_src_homogeneous = np.vstack((xy_src, np.ones((1, xy_src.shape[1]))))

        # Convert to torch tensors
        xy_src_homogeneous_tensor = torch.tensor(xy_src_homogeneous, dtype=torch.float16, device=img_tensor.device)

        # Compute the coordinates in the world frame
        xy_world = torch.inverse(torch.tensor(intrinsic_matrix, dtype=torch.float32, device=img_tensor.device)).to(torch.float16) @ xy_src_homogeneous_tensor
        xy_world = xy_world * depth_tensor.view(1, -1)

        # Compute the coordinates in the new camera frame
        xy_new_cam = torch.inverse(torch.tensor(rotation_matrix, dtype=torch.float32, device=img_tensor.device)).to(torch.float16) @ (xy_world - torch.tensor(translation_vector, dtype=torch.float16, device=img_tensor.device).view(3,1))

        # Compute the coordinates in the new image
        xy_dst_homogeneous = torch.tensor(intrinsic_matrix, dtype=torch.float16, device=img_tensor.device) @ xy_new_cam
        xy_dst = xy_dst_homogeneous[:2, :] / xy_dst_homogeneous[2, :]

        # Reshape to a 2D grid and normalize to [-1, 1]
        xy_dst = xy_dst.reshape(2, h, w)
        xy_dst = (xy_dst - torch.tensor([[w/2.0], [h/2.0]], dtype=torch.float16, device=img_tensor.device).unsqueeze(-1)) / torch.tensor([[w/2.0], [h/2.0]], dtype=torch.float16, device=img_tensor.device).unsqueeze(-1)
        xy_dst = torch.flip(xy_dst, [0])
        xy_dst = xy_dst.permute(1, 2, 0)

        # Perform the warping
        wrapped_img = F.grid_sample(img_tensor, xy_dst.to(torch.float16)[None], align_corners=True, mode='bilinear', padding_mode='reflection')
        wrapped_freq = fft.fftn(wrapped_img, dim=(-2, -1))
        wrapped_freq = fft.fftshift(wrapped_freq, dim=(-2, -1))
        wrapped_img = self.combine_low_and_high(wrapped_freq, high_freq, fft_mask)


        return wrapped_img
    
    
    def wrap_img_tensor(self, img_tensor, depth_tensor, 
                        theta_z=0, theta_x=0, theta_y=-10, T=[0,0,-2]):
        intrinsic_matrix = np.array([[1000, 0, img_tensor.shape[-1]/2],
                    [0, 1000, img_tensor.shape[-2]/2],
                    [0, 0, 1]])  # Example intrinsic matrix
        ori_size = None
        if depth_tensor.shape[-1] != img_tensor.shape[-1]:
            scale = depth_tensor.shape[-1] / img_tensor.shape[-1]
            ori_size = (img_tensor.shape[-2], img_tensor.shape[-1])
            img_tensor_ori = img_tensor.clone()
            # img_tensor = F.interpolate(img_tensor, size=(depth_tensor.shape[-2], depth_tensor.shape[-1]))
            depth_tensor = F.interpolate(depth_tensor.unsqueeze(0).unsqueeze(0), size=ori_size, mode='bilinear').squeeze().to(torch.float16)
            intrinsic_matrix[0,0] /= scale
            intrinsic_matrix[1,1] /= scale
        rotation_matrix = self.calc_R(theta_z=theta_z, theta_x=theta_x, theta_y=theta_y)
        translation_vector = np.array(T)  # Translation vector to shift camera to the right

        h,w = img_tensor.shape[2:]

        xy_src = np.mgrid[0:h, 0:w].reshape(2, -1)

        xy_src_homogeneous = np.vstack((xy_src, np.ones((1, xy_src.shape[1]))))

        # Convert to torch tensors
        xy_src_homogeneous_tensor = torch.tensor(xy_src_homogeneous, dtype=torch.float16, device=img_tensor.device)

        # Compute the coordinates in the world frame
        xy_world = torch.inverse(torch.tensor(intrinsic_matrix, dtype=torch.float32, device=img_tensor.device)).to(torch.float16) @ xy_src_homogeneous_tensor
        xy_world = xy_world * depth_tensor.view(1, -1)

        # Compute the coordinates in the new camera frame
        xy_new_cam = torch.inverse(torch.tensor(rotation_matrix, dtype=torch.float32, device=img_tensor.device)).to(torch.float16) @ (xy_world - torch.tensor(translation_vector, dtype=torch.float16, device=img_tensor.device).view(3,1))

        # Compute the coordinates in the new image
        xy_dst_homogeneous = torch.tensor(intrinsic_matrix, dtype=torch.float16, device=img_tensor.device) @ xy_new_cam
        xy_dst = xy_dst_homogeneous[:2, :] / xy_dst_homogeneous[2, :]

        # Reshape to a 2D grid and normalize to [-1, 1]
        xy_dst = xy_dst.reshape(2, h, w)
        xy_dst = (xy_dst - torch.tensor([[w/2.0], [h/2.0]], dtype=torch.float16, device=img_tensor.device).unsqueeze(-1)) / torch.tensor([[w/2.0], [h/2.0]], dtype=torch.float16, device=img_tensor.device).unsqueeze(-1)
        xy_dst = torch.flip(xy_dst, [0])
        xy_dst = xy_dst.permute(1, 2, 0)

        # Perform the warping
        wrapped_img = F.grid_sample(img_tensor, xy_dst.to(torch.float16)[None], align_corners=True, mode='bilinear')



        return wrapped_img

    @torch.no_grad()
    def __call__(self, img_array, theta_z=0, theta_x=0, theta_y=-10, T=[0,0,-2]):
        img_depth = self.transform({"image": img_array})["image"]

        # compute
        prediction = self.process(
            self.device,
            self.model,
            self.model_type,
            img_depth,
            (self.net_w, self.net_h),
            img_array.shape[1::-1],
            optimize=False,
            use_camera=False,
        )

        depth = self.prediction2depth(prediction)

        # img = img_array.copy()
        # img = img / 2. + 0.5
        K = np.array([[1000, 0, img_array.shape[1]/2],
                    [0, 1000, img_array.shape[0]/2],
                    [0, 0, 1]])  # Example intrinsic matrix

        R = self.calc_R(theta_z=theta_z, theta_x=theta_x, theta_y=theta_y)
        T = np.array(T)  # Translation vector to shift camera to the right

        # new_img = self.render_new_view(img_array, depth, R, T, K)
        new_img = self.wrap_img(img_array, depth, K, R, T)

        mask = np.all(new_img == [0,0,0], axis=2).astype(np.uint8) * 255
        mask = 255 - mask
        return new_img, mask, depth
    
