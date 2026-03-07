# MAMBO-G: Magnitude-Aware Mitigation for Boosted Guidance

(Building now...)

<p align="center">
    <a href="https://github.com/huggingface/diffusers/pull/12862">
        <img src="https://img.shields.io/badge/Official%20Integration-Diffusers-blue?logo=huggingface" alt="Diffusers Integration">
    </a>
    <a href="https://arxiv.org/abs/2508.03442v2">
        <img src="https://img.shields.io/badge/arXiv-2503.09675-b31b1b.svg" alt="arXiv">
    </a>
    <!-- <a href="https://github.com/your-username/MAMBO-G/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-Apache--2.0-green.svg" alt="License">
    </a> -->
</p>

**MAMBO-G** is a **training-free**, universal acceleration framework for Classifier-Free Guidance (CFG). By dynamically optimizing guidance magnitudes based on the update-to-prediction ratio, **MAMBO-G** achieves up to **3.0× speedup** on image models (SD3.5, Lumina, Qwen-Image) and **2.0× speedup** on the Wan2.1-14B video model, all while preserving high visual fidelity.

---

## 🚀 News
- **[2026-02]** :tada: **MAMBO-G has been officially merged into the [Hugging Face Diffusers](https://github.com/huggingface/diffusers) library!** You can now use our method natively via the standard library. [Check PR #12862](https://github.com/huggingface/diffusers/pull/12862).
- **[2025-08]** 🎉 **MAMBO-G now supports [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)!** Achieve state-of-the-art text rendering and image generation with 3x speedup.
- **[2025-08]** Preprint paper is available on [arXiv](https://arxiv.org/abs/2508.03442v2).

---

## 💡 Why MAMBO-G?

During the early steps of the reverse diffusion process, the **relative magnitude** between conditional and unconditional predictions peaks sharply. Standard CFG uses a fixed scale, which leads to:
1. **Trajectory Instability**: High sensitivity in early sampling phases.
2. **Redundant Computation**: Requiring more NFEs (steps) to correct path deviations.

**MAMBO-G** addresses this by modulating the guidance scale adaptive to the magnitude ratio, effectively stabilizing the trajectory and enabling rapid convergence.

<div align="center">
  <img src="figures/head_display.png" alt="MAMBO-G Comparison" width="100%">
  <br>
  <em>Results on SD3.5 and Lumina-Next. Left: 10-step Baseline. Middle: 30-step Baseline. Right: <b>10-step MAMBO-G (Matches 30-step quality with 3x speedup)</b>.</em>
</div>

---

## ✨ Key Features
- **Plug-and-Play**: No training or fine-tuning required. Works with any pre-trained flow-matching model.
- **Extreme Efficiency**: 2x-4x speedup across SD3.5, Lumina, Qwen-Image, and Wan2.1-14B.
- **Official Support**: Natively integrated into the `diffusers` ecosystem.

---

## 🛠️ Quick Start

### Installation
```bash
git clone https://github.com/your-username/MAMBO-G.git
cd MAMBO-G
pip install -r requirements.txt
```

### Usage (Integrated in Diffusers)
With the official integration, accelerating your pipeline is as simple as enabling the `mambo_g_enabled` flag.

#### 1. Stable Diffusion 3.5
```python
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large", 
    torch_dtype=torch.bfloat16
).to("cuda")

# MAMBO-G Accelerated Sampling (Only 10 steps!)
image = pipe(
    "a photo of an astronaut riding a horse on the moon", 
    num_inference_steps=10, 
    guidance_scale=7.0,
    mambo_g_enabled=True,   # Enabled via our official integration
    max_guidance=18.0, 
    lr_para=12.0
).images[0]
```

#### 2. Qwen-Image
```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16).to("cuda")

# MAMBO-G Accelerated Sampling (Only 15 steps for complex text rendering!)
image = pipe(
    prompt="A coffee shop entrance with a sign 'Qwen Coffee 😊 $2 per cup'",
    num_inference_steps=15, 
    mambo_g_enabled=True,
    max_guidance=18.0, 
    lr_para=12.0
).images[0]
```

### Example Scripts
We provide ready-to-use scripts for evaluating MAMBO-G:
- `sd35_sample.py`: Compare Original vs MAMBO-G for SD3.5.
- `qwen_sample.py`: Compare Original vs MAMBO-G for Qwen-Image.
- `qwen_mambo_g.py`: Specialized acceleration script for Qwen-Image.

---

## 📊 Performance Benchmark

| Model | Task | Baseline Steps | MAMBO-G Steps | Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **Stable Diffusion 3.5** | T2I | 30 | 10 | **3.0×** |
| **Qwen-Image** | T2I | 50 | 15 | **3.3×** |
| **Lumina-Next** | T2I | 40 | 10 | **4.0×** |
| **Wan2.1 (14B)** | T2V | 30 | 15 | **2.0×** |

---

## 🖼️ Visual Gallery
*(Optional: Add more high-quality result comparisons here)*

---

## ✒️ Citation

If you find this work helpful, please cite our paper:
```
@article{zhu2025mambo,
  title={MAMBO-G: Magnitude-Aware Mitigation for Boosted Guidance},
  author={Zhu, Shangwen and Peng, Qianyu and Shu, Zhilei and others},
  journal={arXiv preprint arXiv:2503.09675},
  year={2025}
}
```
<p align="center">Developed by the MAMBO-G Team.</p>
