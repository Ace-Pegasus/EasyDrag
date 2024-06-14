from share import *

import os
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import copy
import imageio

from torchvision.utils import save_image
from einops import rearrange
from copy import deepcopy
from types import SimpleNamespace
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked_drag import DDIMSampler
import argparse

parser = argparse.ArgumentParser(description='test at st')
parser.add_argument('--port', type=int)
parser.add_argument('--config', type=str)
parser.add_argument('--ckpt', type=str)
args = parser.parse_args()

config_path = args.config

ckpt_path = args.ckpt

print(config_path, ckpt_path)
model, _ = create_model(config_path)
m, u = model.load_state_dict(load_state_dict(ckpt_path, location='cuda'), strict=False)
if len(m) > 0:
    print('missing:', m)
if len(u) > 0:
    print('unexpected:', u)

model = model.cuda()
# model.half()
ddim_sampler = DDIMSampler(model)

def preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image

def generate_image(prompt, guidance_scale, seed, a_prompt, n_prompt):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # initialize parameters
    seed_everything(seed)
    init_code = torch.randn([1, 4, 64, 64], device=device)

    cond = {"c_concat": None, "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt])]}
    un_cond = {"c_concat": None, "c_crossattn": [model.get_learned_conditioning([n_prompt])]}

    samples, latents_list = ddim_sampler.gen(
            latents=init_code,
            c=cond,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=un_cond,
            return_intermediate=True,
            save_gen=True,
        )
    x_samples = model.decode_first_stage(samples)

    x_sample = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]

    save_sample = preprocess_image(x_sample, device)
    save_sample = save_sample * 0.5 + 0.5

    return latents_list, x_sample


def mask_image(image, mask, color=[255,0,0], alpha=0.5):
    """ Overlay mask on image for visualization purpose. 
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    contours = cv2.findContours(np.uint8(deepcopy(mask)), cv2.RETR_TREE, 
                        cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return out

def inference(source_image,
              image_with_clicks,
              mask,
              prompt,
              points,
              n_actual_inference_step,
              n_pix_step,
              use_gen_img,
              ori_latents_list,
              guidance_scale,
              seed, a_prompt, n_prompt,
              reference_only,
              training_reference,
              ref_start_step,
              neg_same_prompt,
              save_every_step,
              point_track,
              no_mask,
              image_nomask,
              save_dir="./results"
    ):
    # initialize model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # initialize parameters
    seed_everything(seed)
    c_prompt = prompt + ', ' + a_prompt
    if neg_same_prompt:
        c_prompt = prompt
        n_prompt = c_prompt
    cond = {"c_concat": None, "c_crossattn": [model.get_learned_conditioning([c_prompt])]}
    un_cond = {"c_concat": None, "c_crossattn": [model.get_learned_conditioning([n_prompt])]}

    args = SimpleNamespace()
    args.cond = cond
    args.uncond = un_cond
    args.n_inference_step = 50
    args.n_actual_inference_step = n_actual_inference_step
    args.guidance_scale = guidance_scale
    args.reference_only = reference_only
    args.ref_start_step = ref_start_step

    args.unet_feature_idx = [1, 2]

    args.c_prompt = c_prompt
    args.n_prompt = n_prompt

    args.sup_res = 64
    args.r_m = 0

    args.r_p = 3

    args.lr = 0.01

    args.n_pix_step = n_pix_step

    args.training_reference = training_reference
    args.point_track = point_track
    args.no_mask = no_mask

    # print(args)
    full_h, full_w = source_image.shape[:2]
    args.feat_scale = 8
    args.sup_res_h = int(source_image.shape[0] * args.feat_scale / 8)
    args.sup_res_w = int(source_image.shape[1] * args.feat_scale / 8)

    source_image = preprocess_image(source_image, device)
    image_with_clicks = preprocess_image(image_with_clicks, device)
    image_nomask = preprocess_image(image_nomask, device)

    x0 = model.get_first_stage_encoding(model.encode_first_stage(source_image))

    # invert the source image
    # the latent code resolution is too small, only 64*64
    if use_gen_img:
        invert_code = ori_latents_list[n_actual_inference_step]
    else:
        invert_code, ori_latents_list = ddim_sampler.invert(x0,
            cond,
            num_actual_inference_steps=n_actual_inference_step,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=un_cond,
            )

    mask = torch.from_numpy(mask).float()
    mask[mask > 0.0] = 1.0
    if mask.sum == 0.0:
        mask = 1.0 - mask
    mask = rearrange(mask, "h w -> 1 1 h w").cuda()

    handle_points = []
    target_points = []
    ori_handle_points = []
    ori_target_points = []
    # here, the point is in x,y coordinate
    for idx, point in enumerate(points):
        cur_point = torch.tensor([point[1]/full_h *args.sup_res_h, point[0]/full_w*args.sup_res_w])
        # cur_point = torch.tensor([point[1] / 2, point[0] / 2])
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
            ori_handle_points.append([point[1], point[0]])
        else:
            target_points.append(cur_point)
            ori_target_points.append([point[1], point[0]])
    # [40, 32] [40, 39]
    print('handle points:', handle_points)
    print('target points:', target_points)

    # init_code = invert_code
    t = ddim_sampler.ddim_timesteps[n_actual_inference_step-1]
    t = torch.full((x0.shape[0],), t, device=model.device, dtype=torch.long)
    trainable_code = invert_code.clone().detach()

    # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
    # update according to the given supervision
    save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    save_dir = "./results/%s"%save_prefix
    os.makedirs(save_dir, exist_ok=True)
    
    updated_init_code, image_list = drag_diffusion_update(model, trainable_code, t, handle_points, target_points, ori_handle_points, 
                                              ori_target_points, mask, cond, args, save_dir, ori_latents_list)

    # inference the synthesized image
    for scale in [guidance_scale]:
        updated_x0 = ddim_sampler.gen(
                latents=updated_init_code,
                c=cond,
                start_step=n_actual_inference_step,
                end_step=0,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond,
                ori_latents_list=ori_latents_list,
                reference_only=reference_only,
                points=points,
                ref_start_step=ref_start_step,
                save_every_step=save_every_step,
                args=args,
            )

        x_sample = model.decode_first_stage(updated_x0)

        image = (einops.rearrange(x_sample, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().float().numpy().clip(0, 255)[0]
        image = np.ascontiguousarray(image, dtype=np.uint8)
        for i in range(len(handle_points)):
            tmp_p = tuple(handle_points[i].to(torch.int).tolist())
            tmp_p = (tmp_p[1] * 8 // args.feat_scale, tmp_p[0] * 8 // args.feat_scale)
            tmp_t = tuple(target_points[i].to(torch.int).tolist())
            tmp_t = (tmp_t[1] * 8 // args.feat_scale, tmp_t[0] * 8 // args.feat_scale)
            cv2.circle(image, tmp_t, 6, (0, 0, 255), -1)
        image_list.append(image)

        x_sample = (einops.rearrange(x_sample, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]
        x_sample = np.ascontiguousarray(x_sample, dtype=np.uint8)

        # circle target points
        for point in points[1::2]:
            cv2.circle(x_sample, tuple(point), 6, (0, 0, 255), -1)

        # save the original image, user editing instructions, synthesized image
        save_sample = preprocess_image(x_sample, device)
        if args.no_mask:
            save_result = torch.cat([
                source_image * 0.5 + 0.5,
                torch.ones((1,3,full_h,25)).cuda(),
                image_nomask * 0.5 + 0.5,
                torch.ones((1,3,full_h,25)).cuda(),
                save_sample * 0.5 + 0.5,
            ], dim=-1)
        else:
            save_result = torch.cat([
                source_image * 0.5 + 0.5,
                torch.ones((1,3,full_h,25)).cuda(),
                image_with_clicks * 0.5 + 0.5,
                torch.ones((1,3,full_h,25)).cuda(),
                save_sample * 0.5 + 0.5,
            ], dim=-1)

        save_image(save_result, os.path.join(save_dir, 'result%d.png'%scale))

        video_path = os.path.join(save_dir, "output.mp4")
        fps = 2  # 2 frame per second
        with imageio.get_writer(video_path, fps=fps) as video:
            for image in image_list:
                video.append_data(image)

    return x_sample


def get_region(x:torch.Tensor, h0, h1, w0, w1):
    return x[..., max(h0, 0):min(h1, x.shape[-2]), max(w0, 0):min(w1, x.shape[-1])]


def gaussian2d(x, y, sigma=1):
    return np.exp(-(x**2+y**2)/(2*sigma**2))


def point_tracking(F0,
                   F1,
                   handle_points,
                   handle_points_init,
                   target_points,
                   args):
    sim = nn.CosineSimilarity(dim=1)
    update_point = False
    with torch.no_grad():
        for i in range(len(handle_points)):
            pi0, pi, ti = handle_points_init[i], handle_points[i], target_points[i]
            next_pi = next_point(pi, ti)
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]
            f1_ori = F1[:, :, int(pi[0]), int(pi[1])]
            f1 = F1[:, :, int(next_pi[0]), int(next_pi[1])]
            if sim(f0, f1) > sim(f0, f1_ori):
                handle_points[i] = next_pi
                update_point = True
        return handle_points, update_point

def check_handle_reach_target(F0,
                              F1,
                              handle_points,
                              handle_points_init,
                              target_points):
    all_dist = list(map(lambda p,q: (p-q).norm(), handle_points, target_points))
    return (torch.tensor(all_dist) < 1.0).all()


# obtain the bilinear interpolated feature patch centered around (x, y) with radius r
def interpolate_feature_patch(feat,
                              y,
                              x,
                              r):
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    Ia = feat[:, :, y0-r:y0+r+1, x0-r:x0+r+1]
    Ib = feat[:, :, y1-r:y1+r+1, x0-r:x0+r+1]
    Ic = feat[:, :, y0-r:y0+r+1, x1-r:x1+r+1]
    Id = feat[:, :, y1-r:y1+r+1, x1-r:x1+r+1]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd

def next_point(p, t):
    d = (t - p) / (t - p).norm()
    if (t - p).norm() >= 18:
        p_d = torch.round(p + d * 12)
    else:
        p_d = t
    return p_d

def refine_r(p, q, r, shape):
    r_h = min(r, int(p[0]), int(q[0]), shape[-2]-int(p[0]), shape[-2]-int(q[0]))
    r_w = min(r, int(p[1]), int(q[1]), shape[-1]-int(p[1]), shape[-1]-int(q[1]))
    return r_h, r_w

def drag_diffusion_update(model, init_code, t, handle_points, target_points, ori_handle_points, ori_target_points, mask, cond, args, save_dir, ori_code_list=None):

    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"
    
    ori_code = ori_code_list[args.n_actual_inference_step]

    # the init output feature of unet
    with torch.no_grad():
        _, F0 = model.apply_model(init_code, t, cond, return_intermediates=True)
        F0 = F.interpolate(F0, (F0.shape[2] * args.feat_scale, F0.shape[3] * args.feat_scale), mode='bilinear')

    if args.training_reference:
        t = t.repeat(2)
        cond = {"c_concat": None, "c_crossattn": [model.get_learned_conditioning([args.c_prompt]*2)]}
        
    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)

    optimizer = torch.optim.Adam([init_code], lr=args.lr)

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    sim = nn.CosineSimilarity(dim=1)

    handle_points_init = copy.deepcopy(handle_points)

    update_point = False
    update_times = 0

    image_list = []

    for step_idx in range(args.n_pix_step):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if args.training_reference:
                _, F1 = model.apply_model(torch.cat([init_code, ori_code], dim=0), t, cond, return_intermediates=True)
                F1 = F1[:1]
            else:
                _, F1 = model.apply_model(init_code, t, cond, return_intermediates=True)

            F1 = F.interpolate(F1, (F0.shape[2], F0.shape[3]), mode='bilinear')

            loss = 0.0

            if args.point_track:
                handle_points, update_point = point_tracking(F0, F1, handle_points, handle_points_init, target_points, args)
                if step_idx == 0:
                    update_point = True
                if update_point:
                    updated_x0 = ddim_sampler.gen(
                        latents=init_code.clone().detach(),
                        c=args.cond,
                        start_step=args.n_actual_inference_step,
                        end_step=0,
                        unconditional_guidance_scale=args.guidance_scale,
                        unconditional_conditioning=args.uncond,
                        ori_latents_list=ori_code_list,
                        reference_only=args.reference_only,
                        points=handle_points,
                        ref_start_step=args.ref_start_step,
                        args=args,
                    )
                    x_sample = model.decode_first_stage(updated_x0)
                    x_sample = (einops.rearrange(x_sample, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().float().numpy().clip(0, 255)[0]
                    x_sample = np.ascontiguousarray(x_sample, dtype=np.uint8)
                    for i in range(len(handle_points)):
                        tmp_p = tuple(handle_points[i].to(torch.int).tolist())
                        tmp_p = (tmp_p[1] * 8 // args.feat_scale, tmp_p[0] * 8 // args.feat_scale)
                        tmp_t = tuple(target_points[i].to(torch.int).tolist())
                        tmp_t = (tmp_t[1] * 8 // args.feat_scale, tmp_t[0] * 8 // args.feat_scale)
                        cv2.circle(x_sample, tmp_p, 6, (255, 0, 0), -1)
                        cv2.circle(x_sample, tmp_t, 6, (0, 0, 255), -1)
                        cv2.arrowedLine(x_sample, tmp_p, tmp_t, (255, 255, 255), 3, tipLength=0.2)
                    image_list.append(x_sample)
                    save_sample = preprocess_image(x_sample, device=model.device)
                    save_sample = save_sample * 0.5 + 0.5
                    save_image(save_sample, os.path.join(save_dir, 'result_%d.png'%update_times))

            # break if all handle points have reached the targets
            if args.point_track and check_handle_reach_target(F0, F1, handle_points, handle_points_init, target_points):
                break

            for i in range(len(handle_points)):
                pi, ti = handle_points[i], target_points[i]

                # motion supervision
                r_m = args.r_m
                if not args.point_track:
                    f0_patch = get_region(F0, int(pi[0])-r_m, int(pi[0])+r_m+1, int(pi[1])-r_m, int(pi[1])+r_m+1).detach()
                    f1_patch = get_region(F1, int(ti[0])-r_m, int(ti[0])+r_m+1, int(ti[1])-r_m, int(ti[1])+r_m+1)

                else:
                    ori_pi = handle_points_init[i]
                    next_pi = next_point(pi, ti)

                    r_h, r_w = refine_r(ori_pi, next_pi, r_m, F0.shape)
                    f0_patch = F0[:,:,int(ori_pi[0])-r_h:int(ori_pi[0])+r_h+1, int(ori_pi[1])-r_w:int(ori_pi[1])+r_w+1].detach()
                    f1_patch = F1[:,:,int(next_pi[0])-r_h:int(next_pi[0])+r_h+1, int(next_pi[1])-r_w:int(next_pi[1])+r_w+1]

                S_gen = (sim(f1_patch, f0_patch).mean() + 1) / 2
                if torch.equal(pi, ti):
                    L_gen = 0.1 / (1 + 1 * S_gen)
                else:
                    L_gen = 1 / (1 + 1 * S_gen)

                loss += L_gen

            if update_point:
                if args.no_mask:
                    scaler.scale(loss).backward(retain_graph=True)
                    # print(init_code.grad)
                    grad = init_code.grad
                    grad = grad[0].mean(0).abs()
                    min_val = torch.min(grad)
                    max_val = torch.max(grad)
                    grad = 255 * (grad - min_val) / (max_val - min_val)
                    interp_mask = torch.zeros_like(grad)
                    interp_mask[grad>100] = 1

                    grad = np.round(grad.cpu().numpy()).astype(np.uint8)
                    grad_heatmap = cv2.applyColorMap(np.uint8(grad), cv2.COLORMAP_JET)
                    cv2.imwrite('%s/grad_%d.png'%(save_dir, update_times), grad_heatmap)

                    save_mask = interp_mask.cpu().numpy()
                    save_mask = save_mask * 255
                    save_mask = np.round(save_mask).astype(np.uint8)
                    cv2.imwrite('%s/mask_%d.png'%(save_dir, update_times), save_mask)
                    interp_mask = interp_mask.unsqueeze(0).unsqueeze(0)
                    optimizer.zero_grad()

                else:
                    interp_mask = mask

                update_times += 1

            # masked region must stay unchanged
            interp_mask = F.interpolate(interp_mask, (F0.shape[2], F0.shape[3]), mode='nearest')
            if (1.0 - interp_mask).sum() >= 1.:
                f0_share = F0 * (1.0 - interp_mask)
                f1_share = F1 * (1.0 - interp_mask)
                S_share = (sim(f0_share, f1_share).sum() / (1.0 - interp_mask).sum() + 1) / 2
                L_share = 0.1 / (1 + 1 * S_share)
                loss += L_share

            print('loss total=%f'%(loss.item()))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        update_point = False

    return init_code, image_list

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("""
        # Official Implementation of EasyDrag
        """)

    with gr.Tab(label="Image"):
        with gr.Row():
            # input image
            original_image = gr.State(value=None) # store original image
            mask = gr.State(value=None) # store mask
            image_nomask = gr.State(value=None)
            selected_points = gr.State([]) # store points
            latents_list = gr.State(value=None)
            length = 480
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 25px">Draw Mask</p>""")
                canvas = gr.Image(type="numpy", tool="sketch", label="Draw Mask", show_label=True, height=length, width=length) # for mask painting
                # train_lora_button = gr.Button("Train LoRA")
                gen_button = gr.Button("Gen image")
                cut_button = gr.Button("Cut image")
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 25px">Click Points</p>""")
                input_image = gr.Image(type="numpy", label="Click Points", show_label=True, height=length, width=length) # for points clicking
                undo_button = gr.Button("Undo point")
                save_button = gr.Button("Save labels")
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 25px">Editing Results</p>""")
                output_image = gr.Image(type="numpy", label="Editing Results", show_label=True, height=length, width=length)
                run_button = gr.Button("Run")
                batch_run_button = gr.Button("Batch run")


        # general parameters
        with gr.Row():
            prompt = gr.Textbox(label="Prompt")
            a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
            n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')


        # algorithm specific parameters
        with gr.Accordion(label="Algorithm Parameters", open=False):
            with gr.Tab("Drag Parameters"):
                with gr.Row():
                    n_pix_step = gr.Number(value=1000, label="n_pix_step", precision=0)
                    n_actual_inference_step = gr.Number(value=35, label="n_actual_inference_step", precision=0)
                    guidance_scale = gr.Number(value=4, label="guidance scale")
                    ref_start_step = gr.Number(value=20, label="ref start step", precision=0)
                with gr.Row():
                    use_gen_img = gr.Checkbox(value=False, label="based on generated images")
                    neg_same_prompt = gr.Checkbox(value=True, label="neg prompt same as prompt")
                    point_track = gr.Checkbox(value=True, label="point tracking")
                    no_mask = gr.Checkbox(value=True, label="generate mask automaticly")
                    training_reference = gr.Checkbox(value=True, label="reference during training")
                    reference_only = gr.Checkbox(value=True, label="reference during inference")
                    save_every_step = gr.Checkbox(value=False, label="save intermediate results")
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)

    # once user upload an image, the original image is stored in `original_image`
    # the same image is displayed in `input_image` for point clicking purpose
    def store_img(img):
        image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
        image = resize_image(image, 512)
        image = np.array(image)
        mask = resize_image(mask, 512)
        if mask.sum() > 0:
            mask = np.uint8(mask > 0)
            masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
        else:
            mask = np.ones_like(image[:, :, 0], dtype=np.uint8)
            masked_img = image.copy()
        return image, [], masked_img, mask
    canvas.edit(
        store_img,
        [canvas],
        [original_image, selected_points, input_image, mask]
    )

    # user click the image to get points, and show the points on the image
    def get_point(original_image, img, sel_pix, evt: gr.SelectData):
        # collect the selected point
        sel_pix.append(evt.index)
        image_nomask = original_image.copy()
        # draw points
        points = []
        for idx, point in enumerate(sel_pix):
            if idx % 2 == 0:
                # draw a red circle at the handle point
                cv2.circle(img, tuple(point), 6, (255, 0, 0), -1)
                cv2.circle(image_nomask, tuple(point), 6, (255, 0, 0), -1)
            else:
                # draw a blue circle at the target point
                cv2.circle(img, tuple(point), 6, (0, 0, 255), -1)
                cv2.circle(image_nomask, tuple(point), 6, (0, 0, 255), -1)
            points.append(tuple(point))
            # draw an arrow from handle point to target point
            if len(points) == 2:
                cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
                cv2.arrowedLine(image_nomask, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
                points = []
        return image_nomask, img
    input_image.select(
        get_point,
        [original_image, input_image, selected_points],
        [image_nomask, input_image],
    )

    # clear all handle/target points
    def undo_points(original_image, mask):
        if mask.sum() > 0:
            mask = np.uint8(mask > 0)
            masked_img = mask_image(original_image, 1 - mask, color=[0, 0, 0], alpha=0.3)
        else:
            masked_img = original_image.copy()
        return masked_img, []

    undo_button.click(
        undo_points,
        [original_image, mask],
        [input_image, selected_points]
    )

    gen_button.click(
        generate_image,
        [prompt,
        guidance_scale,
        seed,
        a_prompt,
        n_prompt,],
        [latents_list, canvas]
    )

    run_button.click(
        inference,
        [original_image,
        input_image,
        mask,
        prompt,
        selected_points,
        n_actual_inference_step,
        n_pix_step,
        use_gen_img,
        latents_list,
        guidance_scale,
        seed,
        a_prompt,
        n_prompt,
        reference_only,
        training_reference,
        ref_start_step,
        neg_same_prompt,
        save_every_step,
        point_track,
        no_mask,
        image_nomask,
        ],
        [output_image]
    )

b = args.port
block.launch(server_name='0.0.0.0', server_port=b)
