# EasyDrag: Efficient Point-based Manipulation on Diffusion Models (CVPR 2024)

This is a research project, NOT a commercial product.

### Installation

```
conda env create -f environment.yaml
conda activate easydrag
```

### Webui

EasyDrag based on webui

```
python3 webui_drag.py --port 8080 --config models/cldm_drag.yaml --ckpt {sd_pretrained_model_path}
```

#### Parameters

|Parameter|Explanation|
|-----|------|
|n_pix_step|Maximum number of steps of motion supervision. **Increase this if handle points have not been "dragged" to desired position.**|
|n_actual_inference_step|Number of DDIM inversion steps performed.|
|guidance_scale|Guidance scale of reference guidance.|
|ref_start_step|Number of DDIM steps to use reference guidance.|
|use_gen_img|Whether use the generated images.|
|neg_same_prompt|Whether neg prompt is the same as prompt. **It is recommended to be set to True when using real images and False otherwise.**|
|no_mask|Whether use the auto mask generation.|
|reference_only|Whether use reference guidance.|
|save_every_step|Whether save pred_x0 every step when denoising.|

### Pretrained Models

We adopt SD-1.5 as our base model: [SD 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt).

Other fine-tuned variants of SD-1.5 can also be used as base models:

- [Majicmix Realistic V7](https://civitai.com/models/43331/majicmix-realistic)

- [Interior Design Supermix](https://civitai.com/models/28112/xsarchitectural-interiordesign-forxslora)

- [Dvarch](https://civitai.com/models/8552/dvarch-multi-prompt-architecture-tuned-model)

### Acknowledgement

This project is based on [controlnet](https://github.com/lllyasviel/ControlNet). Our EasyDrag UI design is inspired by [DragDiffusion](https://github.com/Yujun-Shi/DragDiffusion). Thanks for their great work!

### Citation
```bibtex
@inproceedings{hou2024easydrag,
  title={EasyDrag: Efficient Point-based Manipulation on Diffusion Models},
  author={Hou, Xingzhong and Liu, Boxiao and Zhang, Yi and Liu, Jihao and Liu, Yu and You, Haihang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8404--8413},
  year={2024}
}
```