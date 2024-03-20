"""SAMPLING ONLY."""

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import datetime
from copy import deepcopy
import torch.nn.functional as nnf
from torch.optim.adam import Adam
from typing import Optional, Union
from einops import rearrange
from torchvision.utils import save_image

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor

POINTS = None
STEP = 50

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    # @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               drag_mode=False,
               trans_pos=None, 
               attention_scale=None, 
               r_2d=None,
               **kwargs
               ):
        global BATCHSIZE
        BATCHSIZE = batch_size

        with open('./print.txt', "w") as f:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            f.write(date + "\n")

        if conditioning is not None:
            if isinstance(conditioning, dict):
                for key in list(conditioning.keys()):
                    if conditioning[key] is not None:
                        break
                ctmp = conditioning[key]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
            print("Initialized with DDIM inversion noises")

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b+1,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            # if self.model.use_original_image:
            #     ref_cur = ref_tensor.clone()
            #     img = torch.cat((img, ref_cur), dim=0)
            # else:
            #     ts_ref = torch.full((ref_tensor.shape[0],), step, device=device, dtype=torch.long)
            #     ref_cur = self.model.q_sample(ref_tensor, ts_ref)
            #     # ratio = np.cos(i/(total_steps-1)*np.pi/2)
            #     # ratio = ratio ** 100
            #     # ratio = 1 if i == 0 else 0
            #     # print(i, 1-ratio, ratio)
            #     # img = img * (1 - ratio) + ref_cur * ratio
            #     img = torch.cat((img, ref_cur), dim=0)
            # ts = torch.cat((ts, ts_ref), dim=0)

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold,)
            img, pred_x0 = outs
            # assert img.shape[0] == 2
            # img, pred_x0 = img[:1], pred_x0[:1]
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None,):
        b, *_, device = *x.shape, x.device

        # global BATCHSIZE
        # if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
        #     model_output = self.model.apply_model(x, t, c)
        # else:
        #     model_t = self.model.apply_model(x, t, c)
        #     model_t = model_t[:BATCHSIZE]

        #     model_uncond = self.model.apply_model(x[:BATCHSIZE], t[:BATCHSIZE], unconditional_conditioning[:BATCHSIZE])
        #     model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        # b = BATCHSIZE
        # x = x[:b]
        model_cond = self.model.apply_model(x, t, c)
        model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
        model_output = model_uncond + unconditional_guidance_scale * (model_cond - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
    
    def null_optimization(self, latents, cond, uncond, num_inner_steps=10, epsilon=1e-5, unconditional_guidance_scale=1.):
        uncond_embeddings = uncond[:BATCHSIZE]
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        timesteps = self.ddim_timesteps
        timesteps = np.flip(timesteps)
        num_ddim_steps = timesteps.shape[0]
        bar = tqdm(total=num_inner_steps * num_ddim_steps)
        device = cond.device
        for i in range(num_ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = torch.full((BATCHSIZE,), timesteps[i], device=device, dtype=torch.long)
            with torch.no_grad():
                noise_pred_cond = self.model.apply_model(latent_cur, t, cond[:BATCHSIZE])
            for j in range(num_inner_steps):
                noise_pred_uncond = self.model.apply_model(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + unconditional_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, latent_cur, num_ddim_steps-i-1)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                print(loss_item)
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings.detach())
            with torch.no_grad():
                noise_pred_cond = self.model.apply_model(latent_cur, t, cond[:BATCHSIZE])
                noise_pred_uncond = self.model.apply_model(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + unconditional_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent_cur = self.prev_step(noise_pred, latent_cur, num_ddim_steps-i-1)
        bar.close()
        return uncond_embeddings_list

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], sample: Union[torch.FloatTensor, np.ndarray], index, use_original_steps=False):
        device = sample.device
        alpha_prod_t = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alpha_prod_t_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        beta_prod_t = 1 - alpha_prod_t
        alpha_prod_t = torch.full((BATCHSIZE, 1, 1, 1), alpha_prod_t[index], device=device)
        alpha_prod_t_prev = torch.full((BATCHSIZE, 1, 1, 1), alpha_prod_t_prev[index], device=device)
        beta_prod_t = torch.full((BATCHSIZE, 1, 1, 1), beta_prod_t[index], device=device)
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, unconditional_guidance_scale=1.0, unconditional_conditioning=None,):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        num_reference_steps = timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        latents = []
        latents.append(x0)
        x_next = x0.clone().detach()
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), timesteps[i], device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c[:BATCHSIZE])
            else:
                assert unconditional_conditioning is not None
                # e_t_uncond, noise_pred = torch.chunk(
                #     self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                #                            torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = self.model.apply_model(x_next, t, c[:BATCHSIZE])
                e_t_uncond = self.model.apply_model(x_next, t, unconditional_conditioning[:BATCHSIZE])
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            latents.append(x_next)

        return x_next, latents

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec

    @torch.no_grad()
    def invert(self, latents, c, num_actual_inference_steps=None, 
               unconditional_guidance_scale=1.0, 
               unconditional_conditioning=None,
               ori_latents_list= None,
               args=None,
               ddim_steps=50, **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        self.make_schedule(ddim_num_steps=ddim_steps, verbose=False)
        timesteps = self.ddim_timesteps
        num_reference_steps = timesteps.shape[0]

        alphas_next = self.ddim_alphas
        alphas = self.ddim_alphas_prev
        betas = 1 - alphas

        latents_list = [latents]

        invert_code = latents.clone().detach()

        for i in tqdm(range(num_reference_steps), desc='DDIM Inversion...'):
            t = torch.full((latents.shape[0],), timesteps[i], device=self.model.device, dtype=torch.long)

            noise_pred = self.model.apply_model(latents, t, c)
            
            latents = latents
            pred_x0 = (latents - betas[i] ** 0.5 * noise_pred) / alphas[i] ** 0.5
            pred_dir = (1 - alphas_next[i]) ** 0.5 * noise_pred
            latents = alphas_next[i] ** 0.5 * pred_x0 + pred_dir
            latents_list.append(latents)

            if i == num_actual_inference_steps - 1:
                invert_code = latents.clone().detach()

        return invert_code, latents_list

    @torch.no_grad()
    def gen(self, latents, c, start_step=50, end_step=0, 
            unconditional_guidance_scale=1.0, 
            unconditional_conditioning=None,
            ori_latents_list=None, 
            return_intermediate=False,
            reference_only=False,
            points=None,
            ref_start_step=40,
            save_every_step=False,
            args=None,
            save_gen=False,
            interp_mask=None,
            ddim_steps=50, **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        global POINTS
        global STEP
        POINTS = points

        self.make_schedule(ddim_num_steps=ddim_steps, verbose=False)
        timesteps = self.ddim_timesteps
        num_reference_steps = timesteps.shape[0]

        alphas_prev = self.ddim_alphas_prev
        alphas = self.ddim_alphas
        betas = 1 - alphas

        if ori_latents_list is not None:
            x0 = ori_latents_list[0]

        latents_list = [latents]

        save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
        save_dir = "./results/%s"%save_prefix
        save_dir2 = "./results/%s_inversion"%save_prefix
        if interp_mask is not None:
            interp_mask = nnf.interpolate(interp_mask, (latents.shape[2], latents.shape[3]), mode='nearest')
        for i in tqdm(reversed(range(num_reference_steps)), desc='Image generation...'):
            if i >= start_step or i < end_step:
                continue

            STEP = i

            t = torch.full((latents.shape[0],), timesteps[i], device=self.model.device, dtype=torch.long)

            if interp_mask is not None:
                latents_ori = ori_latents_list[i]
                latents = latents * interp_mask + latents_ori * (1. - interp_mask)

            if reference_only and i < ref_start_step:
                # latents_ori = self.model.q_sample(x0, t)
                latents_ori = ori_latents_list[i]
                latents = torch.cat([latents, latents_ori], dim=0)
                t = torch.full((latents.shape[0],), timesteps[i], device=self.model.device, dtype=torch.long)
                c = {"c_concat": None, "c_crossattn": [self.model.get_learned_conditioning([args.c_prompt]*2)]}

            # else:
            #     t = torch.full((latents.shape[0],), timesteps[i], device=self.model.device, dtype=torch.long)
            #     c = {"c_concat": None, "c_crossattn": [self.model.get_learned_conditioning([args.c_prompt])]}

            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(latents, t, c)[:1]
            else:
                assert unconditional_conditioning is not None
                noise_pred = self.model.apply_model(latents, t, c)[:1]
                # e_t_uncond = self.model.apply_model(latents[1:], t[1:], unconditional_conditioning)
                # noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)
                e_t_uncond = self.model.apply_model(latents[:1], t[:1], unconditional_conditioning)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)
            
            latents = latents[:1]
            pred_x0 = (latents - betas[i] ** 0.5 * noise_pred) / alphas[i] ** 0.5
            if save_every_step:
                self.save_x0(pred_x0, i, points=points, save_dir=save_dir)
                # latents_ori = ori_latents_list[i]
                # noise_ori = self.model.apply_model(latents_ori, t, c)
                # pred_ori = (latents_ori - betas[i] ** 0.5 * noise_ori) / alphas[i] ** 0.5
                # self.save_x0(pred_ori, i, points=points, save_dir=save_dir2)
            elif save_gen:
                self.save_x0(pred_x0, i, save_dir=save_dir)
            pred_dir = (1 - alphas_prev[i]) ** 0.5 * noise_pred
            latents = alphas_prev[i] ** 0.5 * pred_x0 + pred_dir
            latents_list.append(latents)

        latents_list = list(reversed(latents_list))

        if return_intermediate:
            return latents, latents_list
        else:
            return latents
        
    def save_x0(self, x0, timestep, points=None, save_dir="./results"):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        x_sample = self.model.decode_first_stage(x0)

        x_sample = (rearrange(x_sample, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]
        x_sample = np.ascontiguousarray(x_sample, dtype=np.uint8)

        # circle target points
        # if points is not None:
        #     for point in points[1::2]:
        #         cv2.circle(x_sample, tuple(point), 3, (0, 0, 255), -1)

        cv2.cvtColor(x_sample, cv2.COLOR_RGB2BGR, x_sample)
        cv2.imwrite(os.path.join(save_dir, '%d.png'%timestep), x_sample)
        return