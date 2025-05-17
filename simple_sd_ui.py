import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import time # To provide a unique seed for random generation

# --- Configuration ---
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # You can choose other models from Hugging Face Hub
# MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0" # For SDXL (requires more VRAM)

# Determine the device
if torch.cuda.is_available():
    device = "cuda"
    # For NVIDIA GPUs, float16 can save memory and speed up inference
    torch_dtype = torch.float16
    print("Using CUDA (NVIDIA GPU).")
elif torch.backends.mps.is_available(): # For Apple Silicon (M1/M2 Macs)
    device = "mps"
    torch_dtype = torch.float32 # float16 might have issues on MPS for some models
    print("Using MPS (Apple Silicon GPU).")
else:
    device = "cpu"
    torch_dtype = torch.float32 # float32 is standard for CPU
    print("CUDA or MPS not available. Using CPU (this will be slow).")

# --- Load the Stable Diffusion Pipeline ---
# This will download the model on the first run if it's not already cached.
# This can take a few minutes and several GB of disk space.
print(f"Loading model: {MODEL_ID} onto {device} with dtype {torch_dtype}...")
try:
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch_dtype, use_safetensors=True)
    pipe = pipe.to(device)

    # Optimizations (optional, can help with VRAM on CUDA)
    if device == "cuda":
        # pipe.enable_xformers_memory_efficient_attention() # If xformers is installed
        pipe.enable_attention_slicing() # Reduces VRAM usage at a slight performance cost
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    print("Please ensure you have a stable internet connection if downloading for the first time.")
    print("If you are on a system with limited VRAM, consider using a smaller model or CPU (which will be very slow).")
    exit()

# --- Define the Image Generation Function for Gradio ---
def generate_image(prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, seed_input):
    print(f"\n--- Generating Image ---")
    print(f"Prompt: '{prompt}'")
    if negative_prompt:
        print(f"Negative Prompt: '{negative_prompt}'")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    print(f"Dimensions: {width}x{height}, Seed: {seed_input}")

    # Handle seed
    if seed_input == -1 or seed_input == 0: # Use -1 or 0 for a random seed
        # Using time for a simple random seed, for more robust randomness consider other methods
        seed = int(time.time() * 1000) % (2**32 -1) # Ensure it fits in a 32-bit int range for torch generator
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"Using random seed: {seed}")
    else:
        generator = torch.Generator(device=device).manual_seed(int(seed_input))
        print(f"Using fixed seed: {int(seed_input)}")

    try:
        # Use torch.no_grad() to disable gradient calculations for inference
        with torch.no_grad():
            # For GPU, autocast can provide performance benefits
            if device == "cuda" or device == "mps":
                with torch.autocast(device_type=device): # autocast for cuda or mps
                    image = pipe(
                        prompt,
                        negative_prompt=negative_prompt if negative_prompt else None,
                        num_inference_steps=int(num_inference_steps),
                        guidance_scale=float(guidance_scale),
                        width=int(width),
                        height=int(height),
                        generator=generator
                    ).images[0]
            else: # CPU generation
                 image = pipe(
                    prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    num_inference_steps=int(num_inference_steps), # Consider fewer steps for faster (but lower quality) CPU generation
                    guidance_scale=float(guidance_scale),
                    width=int(width),
                    height=int(height),
                    generator=generator
                ).images[0]
        print("Image generated successfully.")
        return image, f"Seed used: {generator.initial_seed() if generator else 'N/A (check logs)'}"
    except Exception as e:
        print(f"Error during image generation: {e}")
        # Create a placeholder image for error display
        error_img = Image.new('RGB', (width, height), color = (200, 200, 200))
        # Could also add text to this image using PIL.ImageDraw
        return error_img, f"Error: {str(e)}"

# --- Create and Launch the Gradio Interface ---
with gr.Blocks() as iface:
    gr.Markdown("# Simple Stable Diffusion UI (via `diffusers`)")
    gr.Markdown("Enter a prompt and parameters to generate an image locally.")

    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(label="Prompt", value="A majestic lion in a lush jungle, photorealistic", lines=3)
            negative_prompt_input = gr.Textbox(label="Negative Prompt (optional)", placeholder="e.g., ugly, blurry, watermark, text", lines=2)
            with gr.Row():
                steps_input = gr.Slider(minimum=10, maximum=150, step=1, value=25, label="Inference Steps")
                guidance_input = gr.Slider(minimum=1.0, maximum=25.0, step=0.1, value=7.5, label="Guidance Scale")
            with gr.Row():
                width_input = gr.Slider(minimum=256, maximum=1024, step=64, value=512, label="Width")
                height_input = gr.Slider(minimum=256, maximum=1024, step=64, value=512, label="Height")
            seed_val_input = gr.Number(label="Seed (-1 or 0 for random)", value=-1, precision=0)
            generate_button = gr.Button("Generate Image", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image", type="pil")
            seed_output_info = gr.Textbox(label="Generation Info", interactive=False)

    generate_button.click(
        fn=generate_image,
        inputs=[prompt_input, negative_prompt_input, steps_input, guidance_input, width_input, height_input, seed_val_input],
        outputs=[output_image, seed_output_info]
    )

    gr.Markdown("""
    **Notes:**
    * The first image generation after loading the model might be slower.
    * Ensure your `Width` and `Height` are multiples of 64 or 8 for best compatibility with Stable Diffusion models.
    * Higher inference steps generally mean better quality but take longer.
    * Guidance scale controls how much the model adheres to your prompt.
    * If running on CPU, expect significantly longer generation times.
    """)

if __name__ == '__main__':
    print("Launching Gradio interface...")
    # iface.launch(share=True) # To create a temporary public link (use with caution)
    iface.launch()
    print("Gradio interface should be running. Check your terminal for the local URL (usually http://127.0.0.1:7860 or similar).")
