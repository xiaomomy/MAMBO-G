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
images[0].save("./sd35_org.png")

torch.manual_seed(42)
images = pipe(
    prompt, 
    num_inference_steps=inference_steps, 
    guidance_scale = 7,
    choice = True,
    max_guidance = 18, 
    lr_para = 12,  
).images
images[0].save("./sd35_mambo_g.png")
