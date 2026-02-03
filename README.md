# MAMBO-G: Magnitude-Aware Mitigation for Boosted Guidance

(Building Now...)

<p align="center">
  <a href="https://arxiv.org/abs/2508.03442">
    <img src="https://img.shields.io/badge/arXiv-2503.09675-b31b1b.svg">
  </a>
</p>

<div align="center">
  <img src="figures/head_display.png" alt="head_display" width="800">
  <br>
  <em>
      (Results on Stable Diffusion v3.5 and Lumina-Next. Left: 10-step original. Middle: 30-step original. Right: 10-step RAAG.) 
  </em>
</div>
<br>

### Introduction

The **relative strength (RATIO)** between conditional and unconditional noise predictions peaks sharply during the earliest reverse steps, making generation highly sensitive and unstable to guidance scale adjustments in this phase. To address this problem, **RAAG** introduces a RATIO-adaptive strategy that dynamically assigns an appropriate guidance scale at each step, achieving substantial improvements in overall generation quality.


## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```python
import torch
from sd35_step import StableDiffusion3Pipeline
from diffusers import FlowMatchEulerDiscreteScheduler

model_id = "stabilityai/stable-diffusion-3.5-large"
inference_steps = 10

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
device = "cuda"

prompt = "a photo of an astronaut on a moon"

pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    scheduler=scheduler,  
    torch_dtype=torch.bfloat16, 
    timestep_spacing="trailing",
)
pipe = pipe.to(device)

torch.manual_seed(42)
images = pipe(
    prompt, 
    num_inference_steps=inference_steps, 
    guidance_scale = 7,
    choice = False,
).images
images[0].save("./example_org.png")

torch.manual_seed(42)
images = pipe(
    prompt, 
    num_inference_steps=inference_steps, 
    guidance_scale = 7,
    choice = True,
    max_guidance = 18, 
    lr_para = 12,  
).images
images[0].save("./example_RAAG.png")
```

We use the Stable Diffusion v3.5 pipeline as an example, but feel free to customize the `pipe` and `scheduler` as needed.

The remaining LTC-Accel parameters contribute marginally to acceleration efficiency. See `step.py` for implementation details.

## Visualization


