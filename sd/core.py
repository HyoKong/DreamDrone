import torch
import numpy as np
import torch.nn.functional as F
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from typing import Any, Callable, Dict, List, Optional, Union
from sd.pnp_utils import register_time, register_attention_control_efficient_kv_w_mask, register_conv_control_efficient_w_mask
import torch.nn as nn
from sd.dift_sd import MyUNet2DConditionModel, OneStepSDPipeline
import ipdb
from tqdm import tqdm
from lib.midas import MiDas

class DDIMBackward(StableDiffusionPipeline):
    def __init__(
        self, vae, text_encoder, tokenizer, unet, scheduler,
        safety_checker, feature_extractor,
        requires_safety_checker: bool = True,
        device='cuda', model_id='ckpt/stable-diffusion-2-1-base',depth_model='dpt_swin2_large_384'
    ):
        super().__init__(
            vae, text_encoder, tokenizer, unet, scheduler,
            safety_checker, feature_extractor, requires_safety_checker,
        )
        
        self.dift_unet = MyUNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16 if 'cuda' in device else torch.float32)
        self.onestep_pipe =  OneStepSDPipeline.from_pretrained(model_id, unet=self.dift_unet, safety_checker=None, torch_dtype=torch.float16 if 'cuda' in device else torch.float32)
        self.onestep_pipe = self.onestep_pipe.to(device)

        if 'cuda' in device:
            self.onestep_pipe.enable_attention_slicing()
            self.onestep_pipe.enable_xformers_memory_efficient_attention()
        self.ensemble_size = 4
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.midas_model = MiDas(device,model_type=depth_model)

        self.torch_dtype=torch.float16 if 'cuda' in device else torch.float32


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        t_start=None,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # if callback and self.processor:
        #     self.unet.set_attn_processor(self.processor)
        #     self.processor.record = True
        # elif self.processor:
        #     self.processor.record = False

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if t_start and t >= t_start:
                    progress_bar.update()
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # if self.processor:
                #     self.processor.timestep = t.item()

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            image = self.decode_latents(latents)
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            image = self.numpy_to_pil(image)
        else:
            image = self.decode_latents(latents)
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    
    def denoise_w_injection(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        t_start=None,
        attn=0.8,
        f=0.5,
        latent_mask=None,
        guidance_loss_scale=0,
        cfg_decay=False,
        cfg_norm=False,
        lr=1.0,
        up_ft_indexes=[1,2],
        img_tensor=None,
        early_stop=50,
        intrinsic=None, extrinsic=None, threshold=20,depth=None,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat((prompt_embeds[1:], prompt_embeds[1:], prompt_embeds[:1]), dim=0)
        else:
            prompt_embeds = torch.cat([prompt_embeds]*2, dim=0)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # if callback and self.processor:
        #     self.unet.set_attn_processor(self.processor)
        #     self.processor.record = True
        # elif self.processor:
        #     self.processor.record = False

        kv_injection_timesteps = self.scheduler.timesteps[:int(len(self.scheduler.timesteps) * attn)]
        f_injection_timesteps = self.scheduler.timesteps[:int(len(self.scheduler.timesteps) * f)]
        register_attention_control_efficient_kv_w_mask(self, kv_injection_timesteps, mask=latent_mask, do_classifier_free_guidance=do_classifier_free_guidance)
        register_conv_control_efficient_w_mask(self, f_injection_timesteps, mask=latent_mask)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if t_start and t >= t_start:
                    progress_bar.update()
                    continue
                if i > early_stop: guidance_loss_scale = 0 # Early stop (optional)
                # if t > 300: guidance_loss_scale = 0 # Early stop (optional)
                register_time(self, t.item())
                # Set requires grad
                if guidance_loss_scale != 0: 
                    latents = latents.detach().requires_grad_()

                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents    # latents: ori_z + wrap_z
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latent_model_input, latent_model_input[1:]], dim=0)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # if self.processor:
                #     self.processor.timestep = t.item()

                # predict the noise residual
                if guidance_loss_scale != 0:
                    with torch.no_grad():
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample
                else:
                    with torch.no_grad():
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    cfg_scale = guidance_scale
                    if cfg_decay: cfg_scale = 1 + guidance_scale * (1-i/num_inference_steps)
                    noise_pred_text, wrap_noise_pred_text, wrap_noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = wrap_noise_pred_text + cfg_scale * (wrap_noise_pred_text - wrap_noise_pred_uncond)
                else:
                    noise_pred_text, wrap_noise_pred_text = noise_pred.chunk(3)
                    noise_pred = wrap_noise_pred_text

                # Normalize (see https://enzokro.dev/blog/posts/2022-11-15-guidance-expts-1/)
                if cfg_norm:
                    noise_pred = noise_pred * (torch.linalg.norm(wrap_noise_pred_uncond) / torch.linalg.norm(noise_pred))

                if guidance_loss_scale != 0:
                    for up_ft_index in up_ft_indexes:
                        
                        alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        alpha_prod_t_prev = (
                            self.scheduler.alphas_cumprod[timesteps[i - 0]]
                            if i > 0 else self.scheduler.final_alpha_cumprod
                        )

                        mu = alpha_prod_t ** 0.5
                        mu_prev = alpha_prod_t_prev ** 0.5
                        sigma = (1 - alpha_prod_t) ** 0.5
                        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                        pred_x0 = (latents - sigma_prev * noise_pred[:latents.shape[0]]) / mu_prev

                        unet_ft_all = self.onestep_pipe(
                            latents=pred_x0[:1].repeat(self.ensemble_size, 1, 1, 1),
                            t=t,
                            up_ft_indices=[up_ft_index],
                            prompt_embeds=prompt_embeds[:1].repeat(self.ensemble_size, 1, 1)
                        )
                        unet_ft1 = unet_ft_all['up_ft'][up_ft_index].mean(0, keepdim=True) # 1,c,h,w
                        unet_ft1_norm = unet_ft1 / torch.norm(unet_ft1, dim=1, keepdim=True)
                        
                        unet_ft1_norm = self.midas_model.wrap_img_tensor_w_fft_ext(
                            unet_ft1_norm.to(self.torch_dtype),
                            torch.from_numpy(depth).to(device).to(self.torch_dtype), 
                            intrinsic,
                            extrinsic[:3,:3], extrinsic[:3,3], threshold=threshold).to(self.torch_dtype)

                        unet_ft_all = self.onestep_pipe(
                            latents=pred_x0[1:2].repeat(self.ensemble_size, 1, 1, 1),
                            t=t,
                            up_ft_indices=[up_ft_index],
                            prompt_embeds=prompt_embeds[:1].repeat(self.ensemble_size, 1, 1)
                        )
                        unet_ft2 = unet_ft_all['up_ft'][up_ft_index].mean(0, keepdim=True) # 1,c,h,w
                        unet_ft2_norm = unet_ft2 / torch.norm(unet_ft2, dim=1, keepdim=True)
                        c = unet_ft2.shape[1]
                        loss = (-self.cos(unet_ft1_norm.squeeze().view(c, -1).T, unet_ft2_norm.squeeze().view(c, -1).T).mean() + 1) / 2.
                    # Get gradient
                    cond_grad = torch.autograd.grad(loss * guidance_loss_scale, latents)[0][1:2]

                # compute the previous noisy sample x_t -> x_t-1
                noise_pred_ = noise_pred - sigma_prev * cond_grad*lr
                noise_pred_ = torch.cat([noise_pred_text, noise_pred_], dim=0)

                # compute the previous noisy sample x_t -> x_t-1
                with torch.no_grad():
                    latents = self.scheduler.step(noise_pred_, t, latents, **extra_step_kwargs).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            with torch.no_grad():
                image = self.decode_latents(latents)
                image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
                image = self.numpy_to_pil(image)
        else:
            image = self.decode_latents(latents)
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    @torch.no_grad()
    def decoder(self, latents):
        with torch.autocast(device_type=self.device, dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs


    def ddim_inversion_w_grad(self, latent, cond, stop_t, guidance_loss_scale=1.0, lr=1.0):
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type=self.device, dtype=torch.float32):

            for i, t in enumerate(tqdm(timesteps)):
                if t >= stop_t:
                    break

                if guidance_loss_scale != 0: 
                    latent = latent.detach().requires_grad_()
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.onestep_pipe.unet(latent, t, encoder_hidden_states=cond_batch, up_ft_indices=[3], output_eps=True)['eps']
                pred_x0 = (latent - sigma_prev * eps) / mu_prev

                unet_ft_all = self.onestep_pipe(
                                    latents=pred_x0[:1].repeat(self.ensemble_size, 1, 1, 1),
                                    t=t,
                                    up_ft_indices=[1],
                                    prompt_embeds=cond_batch[:1].repeat(self.ensemble_size, 1, 1)
                                )
                unet_ft1 = unet_ft_all['up_ft'][1].mean(0, keepdim=True) # 1,c,h,w
                unet_ft1_norm = unet_ft1 / torch.norm(unet_ft1, dim=1, keepdim=True)

                unet_ft_all = self.onestep_pipe(
                    latents=pred_x0[1:2].repeat(self.ensemble_size, 1, 1, 1),
                    t=t,
                    up_ft_indices=[1],
                    prompt_embeds=cond_batch[:1].repeat(self.ensemble_size, 1, 1)
                )
                unet_ft2 = unet_ft_all['up_ft'][1].mean(0, keepdim=True) # 1,c,h,w
                unet_ft2_norm = unet_ft2 / torch.norm(unet_ft2, dim=1, keepdim=True)
                c = unet_ft2.shape[1]
                loss = (-self.cos(unet_ft1_norm.squeeze().view(c, -1).T.detach(), unet_ft2_norm.squeeze().view(c, -1).T).mean() + 1) / 2.
                print(f'loss: {loss.item()}')
                # Get gradient
                cond_grad = torch.autograd.grad(loss * guidance_loss_scale, latent)[0]

                # latent = latent.detach() - cond_grad  * lr
                latent = mu * pred_x0 + sigma * eps - cond_grad  * lr

        return latent

@torch.no_grad()
def DDPM_forward(x_t_dot, t_start, delta_t, ddpm_scheduler, generator):
    # just simple implementation, this should have an analytical expression
    # TODO: implementation analytical form
    for delta in range(1, delta_t):
        # noise = torch.randn_like(x_t_dot, generator=generator)
        noise = torch.empty_like(x_t_dot).normal_(generator=generator)

        beta = ddpm_scheduler.betas[t_start+delta]
        std_ = beta ** 0.5
        mu_ = ((1 - beta) ** 0.5) * x_t_dot
        x_t_dot = mu_ + std_ * noise
    return x_t_dot

