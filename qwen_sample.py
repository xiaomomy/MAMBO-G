import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.qwenimage import TEXT2IMAGE_BLOCKS
from diffusers.guiders import MagnitudeAwareGuidance, ClassifierFreeGuidance


def main():
    """
    Qwen-Image sampling with and without MAMBO-G (Magnitude-Aware Guidance).
    This script compares standard Classifier-Free Guidance (CFG) against 
    MAMBO-G acceleration for high-fidelity text-to-image generation.
    """

    # 1. Initialize the modular pipeline
    # Qwen-Image uses a specific set of blocks for its modular architecture
    blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)
    
    # Model configuration and loading
    modular_repo_id = "YiYiXu/QwenImage-modular"
    pipeline = blocks.init_pipeline(modular_repo_id)
    
    # Use bfloat16 for improved performance on supported GPUs (e.g., A100, H100)
    # Defaulting to float32 if CUDA is not available, though CUDA is highly recommended
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    pipeline.load_components(torch_dtype=dtype)
    pipeline.to(device)

    # 2. Generation Parameters
    # Using the official comic portrait prompt from the MAMBO-G technical report
    prompt = (
        "a comic potrait of a female necromancer with big and cute eyes, fine - face, "
        "realistic shaded perfect face, fine details. night setting. very anime style. "
        "realistic shaded lighting poster by ilya kuvshinov katsuhiro, magali villeneuve, "
        "artgerm, jeremy lipkin and michael garmash, rob rey and kentaro miura style, "
        "trending on art station"
    )
    width, height = 1328, 1328
    num_inference_steps = 10  # Demonstrating performance at low NFE
    seed = 1

    # 3. Baseline Generation (Standard Classifier-Free Guidance)
    print(f"Running baseline CFG generation (Steps: {num_inference_steps})...")
    # guidance_scale=4.0 is a common stable value for Qwen-Image baseline
    guider_cfg = ClassifierFreeGuidance(guidance_scale=4.0)
    pipeline.update_components(guider=guider_cfg)
    
    generator = torch.Generator(device).manual_seed(seed)
    image_baseline = pipeline(
        prompt=prompt, 
        width=width, 
        height=height, 
        output="images", 
        num_inference_steps=num_inference_steps, 
        generator=generator
    )[0]
    
    filename_org = f"t2v_original_{num_inference_steps}_steps.png"
    image_baseline.save(filename_org)
    print(f"Baseline image saved to: {filename_org}")

    # 4. Accelerated Generation (MAMBO-G)
    print(f"Running MAMBO-G accelerated generation (Steps: {num_inference_steps})...")
    # MAMBO-G parameters: 
    # - guidance_scale: Initial scale (boosted)
    # - alpha: Suppression rate for early-step overshoot
    # - guidance_rescale: Normalizes the noise magnitude
    guider_mambo = MagnitudeAwareGuidance(
        guidance_scale=10.0, 
        alpha=8.0, 
        guidance_rescale=1.0
    )
    pipeline.update_components(guider=guider_mambo)
    
    generator = torch.Generator(device).manual_seed(seed)
    image_mambo = pipeline(
        prompt=prompt, 
        width=width, 
        height=height, 
        output="images", 
        num_inference_steps=num_inference_steps, 
        generator=generator
    )[0]
    
    filename_mambo = f"t2v_mambo_{num_inference_steps}_steps.png"
    image_mambo.save(filename_mambo)
    print(f"MAMBO-G image saved to: {filename_mambo}")


if __name__ == "__main__":
    main()
