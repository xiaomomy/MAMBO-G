import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.qwenimage import TEXT2IMAGE_BLOCKS
from diffusers.guiders import MagnitudeAwareGuidance, ClassifierFreeGuidance

MODEL_REPO = "YiYiXu/QwenImage-modular"
OUTPUT_SIZE = (1328, 1328)
STEPS = 10


def build_pipeline(device: str, dtype):
    """Construct the pipeline and send it to the target device."""
    blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)
    pipeline = blocks.init_pipeline(MODEL_REPO)
    pipeline.load_components(torch_dtype=dtype)
    return pipeline.to(device)


def generate_image(
    pipeline,
    device: str,
    prompt: str,
    steps: int,
    mambo_g_enabled: bool,
):
    """Run a single inference pass with or without MAMBO-G enabled."""
    generator = torch.Generator(device=device).manual_seed(42)
    
    if mambo_g_enabled:
        guider = MagnitudeAwareGuidance(guidance_scale=10.0, alpha=8.0, guidance_rescale=1.0)
    else:
        guider = ClassifierFreeGuidance(guidance_scale=4.0)
        
    pipeline.update_components(guider=guider)

    return pipeline(
        prompt=prompt,
        width=OUTPUT_SIZE[0],
        height=OUTPUT_SIZE[1],
        output="images",
        num_inference_steps=steps,
        generator=generator,
    )[0]


def main():
    """Invoke both baseline and MAMBO-G generations for comparison."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipeline = build_pipeline(device, dtype)

    base_prompt = (
        'A coffee shop entrance features a chalkboard sign reading '
        '"Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying '
        '"通义千问". Next to it hangs a poster showing a beautiful Chinese '
        'woman, and beneath the poster is written '
        '"π≈3.1415926-53589793-23846264-33832795-02384197".'
    )
    full_prompt = base_prompt + ", Ultra HD, 4K, cinematic composition."

    print("Running baseline generation (Original method).")
    baseline_image = generate_image(
        pipeline,
        device,
        full_prompt,
        STEPS,
        mambo_g_enabled=False,
    )
    baseline_image.save("qwen_org.png")

    print("Running MAMBO-G accelerated generation.")
    mambo_image = generate_image(
        pipeline,
        device,
        full_prompt,
        STEPS,
        mambo_g_enabled=True,
    )
    mambo_image.save("qwen_mambo_g.png")

    print("Generation complete! Results saved as 'qwen_org.png' and 'qwen_mambo_g.png'.")


if __name__ == "__main__":
    main()
