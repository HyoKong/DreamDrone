import gradio as gr
import torch
import numpy as np

import sd.gradio_utils as gradio_utils

import os
import cv2
import argparse
import ipdb

import argparse
from tqdm import tqdm
from diffusers import DDIMScheduler
from diffusers import  DDIMScheduler, DDPMScheduler

from sd.core import DDIMBackward, DDPM_forward

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def slerp(R_target, rotation_speed):
    # Compute the angle of rotation from the rotation matrix
    angle = np.arccos((np.trace(R_target) - 1) / 2)

    # Handle the case where angle is very small (no significant rotation)
    if angle < 1e-6:
        return np.eye(3)

    # Normalize the angle based on rotation_speed
    normalized_angle = angle * rotation_speed

    # Axis of rotation
    axis = np.array([R_target[2, 1] - R_target[1, 2], 
                     R_target[0, 2] - R_target[2, 0], 
                     R_target[1, 0] - R_target[0, 1]])
    axis = axis / np.linalg.norm(axis)

    # Return the interpolated rotation matrix
    return cv2.Rodrigues(axis * normalized_angle)[0]


def compute_extrinsic_parameters(clicked_point, depth, intrinsic_matrix, rotation_speed, step_x=0, step_y=0, step_z=0):
    # Normalize the clicked point
    x,y = clicked_point
    x = int(x)
    y = int(y)
    x_normalized = (x - intrinsic_matrix[0, 2]) / intrinsic_matrix[0, 0]
    y_normalized = (y - intrinsic_matrix[1, 2]) / intrinsic_matrix[1, 1]

    # Depth at the clicked point
    try:
        z = depth[y, x]
    except Exception:
        ipdb.set_trace()

    # Direction vector in camera coordinates
    direction_vector = np.array([x_normalized * z, y_normalized * z, z])

    # Calculate rotation angles to bring the clicked point to the center
    angle_y = -np.arctan2(direction_vector[1], direction_vector[2])  # Rotation about Y-axis
    angle_x = np.arctan2(direction_vector[0], direction_vector[2])  # Rotation about X-axis

    # Apply rotation speed
    angle_y *= rotation_speed
    angle_x *= rotation_speed

    # Compute rotation matrices
    R_x = cv2.Rodrigues(np.array([1, 0, 0]) * angle_x)[0]
    R_y = cv2.Rodrigues(np.array([0, 1, 0]) * angle_y)[0]
    R = R_y @ R_x

    # Compute rotation matrix to align direction vector with principal axis
    T = np.array([step_x, -step_y, -step_z])

    # Create extrinsic matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = R
    extrinsic_matrix[:3, 3] = T

    return extrinsic_matrix

@torch.no_grad()
def encode_imgs(imgs):
    imgs = 2 * imgs - 1
    posterior = pipe.vae.encode(imgs).latent_dist
    latents = posterior.mean * 0.18215
    return latents

@torch.no_grad()
def decode_latents(latents):
    latents = 1 / 0.18215 * latents
    imgs = pipe.vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    return imgs

@torch.no_grad()
def ddim_inversion(latent, cond, stop_t=1000, start_t=-1):
    timesteps = reversed(pipe.scheduler.timesteps)
    pipe.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(timesteps)):
        if t >= stop_t:
            break
        if t <=start_t:
            continue
        cond_batch = cond.repeat(latent.shape[0], 1, 1)

        alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            pipe.scheduler.alphas_cumprod[timesteps[i - 1]]
            if i > 0 else pipe.scheduler.final_alpha_cumprod
        )

        mu = alpha_prod_t ** 0.5
        mu_prev = alpha_prod_t_prev ** 0.5
        sigma = (1 - alpha_prod_t) ** 0.5
        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

        eps = pipe.unet(latent, t, encoder_hidden_states=cond_batch).sample

        pred_x0 = (latent - sigma_prev * eps) / mu_prev
        latent = mu * pred_x0 + sigma * eps

    return latent

@torch.no_grad()
def get_text_embeds(prompt, negative_prompt='', batch_size=1):
    text_input = pipe.tokenizer(prompt, padding='max_length', max_length=77, truncation=True, return_tensors='pt')
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

    uncond_input = pipe.tokenizer(negative_prompt, padding='max_length', max_length=77, truncation=True, return_tensors='pt')
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

    # cat for final embeddings
    text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size).to(torch_dtype)
    return text_embeddings

def save_video(frames, fps=10, out_path='output/output.mp4'):
    video_dims = (512, 512)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')    
    video = cv2.VideoWriter(out_path,fourcc, fps, video_dims)
    for frame in frames:
        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video.release()

def draw_prompt(prompt):
    return prompt

def to_image(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    arr = tensor.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr * 255
    return arr.astype('uint8')

def add_points_to_image(image, points):
    image = gradio_utils.draw_handle_target_points(image, points, 5)
    return image


def on_click(state, seed, count, prompt, neg_prompt, speed_r, speed_x, speed_y, speed_z, t1, t2, t3, lr, guidance_weight,attn,threshold, early_stop, evt: gr.SelectData):
    end_id = int(t1)
    start_id=int(t2)
    startstart_id = int(t3) 
    timesteps = reversed(ddim_scheduler.timesteps)
    end_t = timesteps[end_id]
    start_t = timesteps[start_id]
    startstart_t = timesteps[startstart_id]
    attn=float(attn)
    cfg_norm=False
    cfg_decay=False
    guidance_loss_scale = float(guidance_weight)
    lr = float(lr)
    threshold = int(threshold)
    up_ft_indexes = 2
    early_stop = int(early_stop)
    generator = torch.Generator(device).manual_seed(int(seed))   # 19491001

    state['direction_offset'] = [int(evt.index[0]), int(evt.index[1])]
    cond = pipe._encode_prompt(prompt, device, 1, True, '')
    for _ in range(int(count)):
        image = state['img']
        img_tensor = torch.from_numpy(np.array(image) / 255.).to(device).to(torch_dtype).permute(2,0,1).unsqueeze(0)
        _,_,depth = pipe.midas_model(np.array(image))

        centered = is_centered(state['direction_offset'])
        if centered:
            extrinsic = compute_extrinsic_parameters(state['direction_offset'], depth, intrinsic, rotation_speed=float(0), step_z=float(speed_z), step_x=float(speed_x), step_y=float(speed_y))
            state['centered'] = centered
        else:
            extrinsic = compute_extrinsic_parameters(state['direction_offset'], depth, intrinsic, rotation_speed=float(speed_r), step_z=float(speed_z), step_x=float(speed_x), step_y=float(speed_y))

        this_latent = encode_imgs(img_tensor)
        this_ddim_inv_noise_end = ddim_inversion(this_latent, cond[1:], stop_t=end_t)
        this_ddim_inv_noise_start = ddim_inversion(this_latent, cond[1:], stop_t=startstart_t)
        
        wrapped_this_ddim_inv_noise_end = pipe.midas_model.wrap_img_tensor_w_fft_ext(this_ddim_inv_noise_end.to(torch_dtype),
                                                                                    torch.from_numpy(depth).to(device).to(torch_dtype), 
                                                                                    intrinsic,
                                                                                    extrinsic[:3,:3], extrinsic[:3,3], threshold=threshold).to(torch_dtype)
        
        wrapped_this_ddim_inv_noise_start = ddim_inversion(wrapped_this_ddim_inv_noise_end, cond[1:], stop_t=start_t, start_t=end_t,)
        wrapped_this_ddim_inv_noise_start = DDPM_forward(wrapped_this_ddim_inv_noise_start, t_start=start_t, delta_t=(startstart_id-start_id)*20, 
                                                        ddpm_scheduler=ddpm_scheduler, generator=generator)
        
        new_img = pipe.denoise_w_injection(
                prompt, generator=generator, num_inference_steps=num_inference_steps,
                latents=torch.cat([this_ddim_inv_noise_start, wrapped_this_ddim_inv_noise_start], dim=0), t_start=startstart_t, 
                latent_mask=torch.ones_like(this_latent[0,0,...], device=device,
                ).unsqueeze(0),
                f=0, attn=attn, guidance_scale=7.5, negative_prompt=neg_prompt,
                guidance_loss_scale=guidance_loss_scale, early_stop=early_stop, up_ft_indexes=[up_ft_indexes],
                cfg_norm=cfg_norm, cfg_decay=cfg_decay, lr=lr, 
                intrinsic=intrinsic, extrinsic=extrinsic, threshold=threshold,depth=depth,
            ).images[1]

        new_img = np.array(new_img).astype(np.uint8)
        state['img'] = new_img

        state['img_his'].append(new_img)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 1.
        state['depth_his'].append(depth)

    return new_img, depth, state['img_his'], state

def is_centered(clicked_point, image_dimensions=(512, 512), threshold=5):
    image_center = [dim // 2 for dim in image_dimensions]
    return all(abs(clicked_point[i] - image_center[i]) <= threshold for i in range(2))


def gen_img(prompt, neg_prompt, state, seed):
    generator = torch.Generator(device).manual_seed(int(seed))   # 19491001
    img = pipe(
        prompt, generator=generator, num_inference_steps=num_inference_steps, negative_prompt=neg_prompt,
    ).images[0]
    img_array = np.array(img)
    _,_,depth = pipe.midas_model(img_array)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 1.

    state['img_his'] = [img_array]
    state['depth_his'] = [depth]
    try:
        state['ori_img'] = img_array
        state['img'] = img_array
    except Exception:
        ipdb.set_trace()
    return img_array, depth, [img_array], state

def on_undo(state):
    if len(state['img_his'])>1:
        del state['img_his'][-1]
        del state['depth_his'][-1]
        image = state['img_his'][-1]
        depth = state['depth_his'][-1]
    else:
        image = state['img_his'][-1]
        depth = state['depth_his'][-1]
    state['img'] = image
    return image, depth, state['img_his'], state

def on_reset(state):
    image = state['img_his'][0]
    depth = state['depth_his'][0]
    state['img'] = image
    state['img_his'] = [image]
    state['depth_his'] = [depth]
    return image, depth, state['img_his'], state

def get_prompt(text):
    return text

def on_save(state, video_name):
    save_video(state['img_his'], fps=5, out_path=f'output/{video_name}.mp4')

def on_seed(seed):
    return int(seed)

def main(args):
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # DreamDrone

            Official implementation of [DreamDrone](https://hyokong.github.io/publications/dreamdrone-page/).

            ## Tutorial

            1. Enter your prompt (and a negative prompt, if necessary) in the textbox, then click the `Generate first image` button.
            2. Adjust the camera's moving speed in the `Direction` panel and set hyperparameters in the `Hyper params` panel.
            3. Click on the generated image to make the camera fly towards the clicked direction. 
            4. The generated images will be displayed in the gallery at the bottom. You can view these images by clicking on them in the gallery or by using the left/right arrow buttons.

            ## Hints

            - You can set the number of images to generate after clicking on an image, for convenience.
            - Our system uses a right-hand coordinate system, with the Z-axis pointing into the image.
            - The rotation speed determines how quickly the camera moves towards the clicked direction (rotation only, no translation). Increase this if you need faster camera pose changes.
            - The Speed XYZ-axis controls the camera's movement along the X, Y, and Z axes. Adjust these parameters for different movement styles, similar to a camera arm.
            - $t_1$ represents the timestep that wraps the latent code.
            - Noise is added from $t_1$ to $t_3$. Between $t_1$ and $t_2$, noise is sourced from a pretrained diffusion U-Net. From $t_2$ to $t_3$, random Gaussian noise is used.
            - The `Learning rate` and `Feature Correspondence Guidance` control the feature-correspondence guidance weight during the denoising process (from timestep $t_3$ to $0$).
            - The `KV injection` parameter adjusts the extent of key and value injection from the current frame to the next.

            > If you encounter any problems, please open an issue. Also, don't forget to star the [Official Github Repo](https://github.com/HyoKong/DreamDrone).

            ***Without further ado, welcome to DreamDrone – enjoy piloting your virtual drone through imaginative landscapes!***


            """,
        )
        img = np.zeros((512, 512, 3)).astype(np.uint8)
        depth_img = np.zeros((512, 512, 3)).astype(np.uint8)
        intrinsic_matrix = np.array([[1000, 0, 512/2],
                    [0, 1000, 512/2],
                    [0, 0, 1]])  # Example intrinsic matrix
        extrinsic_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0]],
                                    dtype=np.float32)
        direction_offset = (255, 255)
        state = gr.State({
            'ori_img': img,
            'img': None,
            'centered': False,
            'img_his': [],
            'depth_his': [],
            'intrinsic': intrinsic_matrix,
            'extrinsic': extrinsic_matrix,
            'direction_offset': direction_offset
        })
        
        with gr.Row():
            with gr.Column(scale=0.2):
                with gr.Accordion("Direction"):
                    speed_r = gr.Number(value=0.05, label='Rotation Speed', step=0.01, minimum=0, maximum=1)
                    speed_x = gr.Number(value=0, label='Speed X-axis', step=1, minimum=-10, maximum=20.0)
                    speed_y = gr.Number(value=0, label='Speed Y-axis', step=1, minimum=-10, maximum=20.0)
                    speed_z = gr.Number(value=5, label='Speed Z-axis', step=1, minimum=-10, maximum=20.0)
                with gr.Accordion('Hyper params'):
                    with gr.Row():
                        count = gr.Number(value=5, label='Num. of generated images', step=1, minimum=1, maximum=10, precision=0)
                        seed = gr.Number(value=19491000, label='Seed', precision=0)
                        t1 = gr.Slider(1, 49, 2, step=1, label='t1')
                        t2 = gr.Slider(1, 49, 12, step=1, label='t2')
                        t3 = gr.Slider(1, 49, 27, step=1, label='t3')
                        lr = gr.Slider(0, 500, 300, step=1, label='Learning rate')
                        guidance_weight = gr.Slider(0, 10, 0.1, step=0.1, label='Feature correspondance guidance')
                        attn = gr.Slider(0, 1, 0.5, step=0.1, label='KV injection')
                        threshold = gr.Slider(0, 31, 20, step=1, label='Threshold of low-pass filter')
                        early_stop = gr.Slider(0, 50, 48, step=1, label='Early stop timestep for feature-correspondance guidance')
                        video_name = gr.Textbox(
                            label="Saved video name", show_label=True, max_lines=1, placeholder='saved video name', value='output',
                        ).style()

            with gr.Column():
                with gr.Box():
                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                        text = gr.Textbox(
                            label="Enter your prompt", show_label=False, max_lines=1, placeholder='Enter your prompt', value='Backyards of Old Houses in Antwerp in the Snow, van Gogh',
                        ).style(
                            border=(True, False, True, True),
                            rounded=(True, False, False, True),
                            container=False,
                        )
                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                        with gr.Column(scale=0.8):
                            neg_text = gr.Textbox(
                                label="Enter your negative prompt", show_label=False, max_lines=1, value='', placeholder='Enter your negative prompt',
                            ).style(
                                border=(True, False, True, True),
                                rounded=(True, False, False, True),
                                container=False,
                            )
                        with gr.Column(scale=0.2):
                            gen_btn = gr.Button("Generate first image").style(
                                margin=False,
                                rounded=(False, True, True, False),
                            )

                with gr.Box():
                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                        with gr.Column():
                            with gr.Tab('Current view'):
                                image = gr.Image(img).style(height=600, width=600)
                        with gr.Column():
                            with gr.Tab('Depth'):
                                depth_image = gr.Image(depth_img).style(height=600, width=600)
                with gr.Row():
                        with gr.Column(min_width=100):
                            reset_btn = gr.Button('Clear All')
                        with gr.Column(min_width=100):
                            undo_btn = gr.Button('Undo Last')
                        with gr.Column(min_width=100):
                            save_btn = gr.Button('Save Video')
                with gr.Row():
                    with gr.Tab('Generated image gallery'):
                        gallery = gr.Gallery(
                            label='Generated images', show_label=False, elem_id='gallery', preview=True, rows=1, height=368,
                        ).style()

        image.select(on_click, [state, seed, count, text, neg_text, speed_r, speed_x, speed_y, speed_z, t1, t2, t3, lr, guidance_weight,attn,threshold, early_stop], [image, depth_image, gallery, state])
        text.submit(get_prompt, inputs=[text], outputs=[text])
        neg_text.submit(get_prompt, inputs=[neg_text], outputs=[neg_text])
        gen_btn.click(gen_img, inputs=[text, neg_text, state, seed], outputs=[image, depth_image, gallery, state])
        reset_btn.click(on_reset, inputs=[state], outputs=[image, depth_image, gallery, state])
        undo_btn.click(on_undo, inputs=[state], outputs=[image, depth_image, gallery, state])
        save_btn.click(on_save, inputs=[state, video_name], outputs=[])

        global num_inference_steps
        global pipe
        global intrinsic
        global ddim_scheduler
        global ddpm_scheduler
        global device
        global model_id
        global torch_dtype

        num_inference_steps = 50

        device = args.device
        model_id = args.model_id
        ddim_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        ddpm_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        torch_dtype=torch.float16 if 'cuda' in str(device) else torch.float32

        pipe = DDIMBackward.from_pretrained(
            model_id, scheduler=ddim_scheduler, torch_dtype=torch_dtype,
            cache_dir='.', device=str(device), model_id=model_id, depth_model=args.depth_model,
        ).to(str(device))
        
        if 'cuda' in str(device):
            pipe.enable_attention_slicing()
            pipe.enable_xformers_memory_efficient_attention()

        intrinsic = np.array([[1000, 0, 256],
                [0, 1000., 256],
                [0, 0, 1]])  # Example intrinsic matrix
    return demo


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model_id', default='ckpt/stable-diffusion-2-1-base')
    parser.add_argument('--depth_model', default='dpt_beit_large_512', choices=['dpt_beit_large_512', 'dpt_swin2_large_384'])
    parser.add_argument('--share', action='store_true')
    parser.add_argument('-p', '--port', type=int, default=None)
    parser.add_argument('--ip', default=None)
    args = parser.parse_args()
    demo = main(args)
    print('Successfully loaded, starting gradio demo')
    demo.queue(concurrency_count=1, max_size=20).launch(share=args.share, server_name=args.ip, server_port=args.port)
