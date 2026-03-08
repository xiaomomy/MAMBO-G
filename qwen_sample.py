import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.qwenimage import TEXT2IMAGE_BLOCKS
from diffusers.guiders import MagnitudeAwareGuidance, ClassifierFreeGuidance

# Model configuration
MODEL_REPO_ID = "YiYiXu/QwenImage-modular"
OUTPUT_SIZE = (1328, 1328)
STEPS = 10
SEED = 1


def build_pipeline(device: str, dtype: torch.dtype):
    """
    Construct the modular pipeline for Qwen-Image.
    
    Args:
        device (str): Device to run the model on (e.g., 'cuda').
        dtype (torch.dtype): Torch data type for model weights.
    
    Returns:
        The initialized pipeline.
    """
    blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)
    pipeline = blocks.init_pipeline(MODEL_REPO_ID)
    pipeline.load_components(torch_dtype=dtype)
    return pipeline.to(device)


def generate_and_save(
    pipeline,
    device: str,
    prompt: str,
    steps: int,
    mambo_g_enabled: bool,
    filename: str,
):
    """
    Execute inference with specified guidance and save the resulting image.
    
    Args:
        pipeline: The initialized pipeline object.
        device (str): Inference device.
        prompt (str): Text prompt for generation.
        steps (int): Number of inference steps.
        mambo_g_enabled (bool): Whether to use Magnitude-Aware Guidance.
        filename (str): Path to save the output image.
    """
    generator = torch.Generator(device).manual_seed(SEED)
    
    # Select guider based on method
    if mambo_g_enabled:
        # Optimized parameters for MAMBO-G
        guider = MagnitudeAwareGuidance(
            guidance_scale=10.0, 
            alpha=8.0, 
            guidance_rescale=1.0
        )
    else:
        # Standard Classifier-Free Guidance
        guider = ClassifierFreeGuidance(guidance_scale=4.0)
        
    pipeline.update_components(guider=guider)

    # Run inference
    image = pipeline(
        prompt=prompt,
        width=OUTPUT_SIZE[0],
        height=OUTPUT_SIZE[1],
        output="images",
        num_inference_steps=steps,
        generator=generator,
    )[0]
    
    image.save(filename)


def main():
    """
    Compare Baseline vs MAMBO-G on Qwen-Image using the official comic portrait prompt.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipeline = build_pipeline(device, dtype)

    # Official prompt from the blog example
    prompt = (
        "a comic potrait of a female necromancer with big and cute eyes, fine - face, "
        "realistic shaded perfect face, fine details. night setting. very anime style. "
        "realistic shaded lighting poster by ilya kuvshinov katsuhiro, magali villeneuve, "
        "artgerm, jeremy lipkin and michael garmash, rob rey and kentaro miura style, "
        "trending on art station"
    )

    # 1. Baseline Generation (Original method)
    print(f"Running baseline generation ({STEPS} steps)...")
    generate_and_save(
        pipeline,
        device,
        prompt,
        STEPS,
        mambo_g_enabled=False,
        filename=f"t2v_original_{STEPS}_steps.png"
    )

    # 2. MAMBO-G Generation (Accelerated method)
    print(f"Running MAMBO-G accelerated generation ({STEPS} steps)...")
    generate_and_save(
        pipeline,
        device,
        prompt,
        STEPS,
        mambo_g_enabled=True,
        filename=f"t2v_mambo_{STEPS}_steps.png"
    )

    print(f"Generation complete! Results saved as 't2v_original_{STEPS}_steps.png' "
          f"and 't2v_mambo_{STEPS}_steps.png'.")


if __name__ == "__main__":
    main()
