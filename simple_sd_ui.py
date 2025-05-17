import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import time
import os

# --- Configuration ---
DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"  # Fallback model
CUSTOM_MODEL_DIR = "/teamspace/studios/this_studio/SD_models"
ALLOWED_EXTENSIONS = (".safetensors",)

# Determine the device
if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
    print("Using CUDA (NVIDIA GPU).")
elif torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float32
    print("Using MPS (Apple Silicon GPU).")
else:
    device = "cpu"
    torch_dtype = torch.float32
    print("CUDA or MPS not available. Using CPU (this will be slow).")

# --- Helper Function to Find Available Models ---
def get_available_models(model_dir):
    models = []
    if os.path.isdir(model_dir):
        for item in os.listdir(model_dir):
            if item.endswith(ALLOWED_EXTENSIONS):
                models.append(os.path.join(model_dir, item))
            elif os.path.isdir(os.path.join(model_dir, item)):
                # Consider adding a recursive search if you have models in subdirectories
                pass
    return ["Default (Hugging Face)"] + models

# Get the initial list of available models
available_models = get_available_models(CUSTOM_MODEL_DIR)

# --- Load the Stable Diffusion Pipeline ---
def load_pipeline(model_path):
    print(f"Loading model from: {model_path} onto {device} with dtype {torch_dtype}...")
    try:
        if model_path == "Default (Hugging Face)":
            pipe = StableDiffusionPipeline.from_pretrained(DEFAULT_MODEL_ID, torch_dtype=torch_dtype, use_safetensors=True)
        elif os.path.isfile(model_path) and model_path.endswith(".safetensors"):
            pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch_dtype, use_safetensors=True)
        elif os.path.isdir(model_path):
            pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype, use_safetensors=True)
        else:
            raise FileNotFoundError(f"Model file or directory not found at: {model_path}")

        pipe = pipe.to(device)
        if device == "cuda":
            pipe.enable_attention_slicing()
        print("Model loaded successfully.")
        return pipe
    except Exception as e:
        print(f"Error loading the model: {e}")
        print(f"Please ensure the path is correct or that the default model can be downloaded.")
        return None

# Load the default pipeline initially
pipeline = load_pipeline("Default (Hugging Face)")

# --- Define the Image Generation Function for Gradio ---
def generate_image(prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, seed_input, selected_model):
    global pipeline # To update the pipeline if a new model is selected

    new_pipeline = load_pipeline(selected_model)
    if new_pipeline:
        pipeline = new_pipeline

    if pipeline is None:
        return Image.new('RGB', (width, height), color = (255, 0, 0)), "Error: Stable Diffusion pipeline not loaded."

    print(f"\n--- Generating Image ---")
    print(f"Prompt: '{prompt}'")
    if negative_prompt:
        print(f"Negative Prompt: '{negative_prompt}'")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    print(f"Dimensions: {width}x{height}, Seed: {seed_input}")
    print(f"Selected Model: '{selected_model}'")

    if seed_input == -1 or seed_input == 0:
        seed = int(time.time() * 1000) % (2**32 -1)
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"Using random seed: {seed}")
    else:
        generator = torch.Generator(device=device).manual_seed(int(seed_input))
        print(f"Using fixed seed: {int(seed_input)}")

    try:
        with torch.no_grad():
            if device == "cuda" or device == "mps":
                with torch.autocast(device_type=device):
                    image = pipeline(
                        prompt,
                        negative_prompt=negative_prompt if negative_prompt else None,
                        num_inference_steps=int(num_inference_steps),
                        guidance_scale=float(guidance_scale),
                        width=int(width),
                        height=int(height),
                        generator=generator
                    ).images[0]
            else:
                image = pipeline(
                    prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(guidance_scale),
                    width=int(width),
                    height=int(height),
                    generator=generator
                ).images[0]
        print("Image generated successfully.")
        return image, f"Seed used: {generator.initial_seed() if generator else 'N/A'}"
    except Exception as e:
        print(f"Error during image generation: {e}")
        error_img = Image.new('RGB', (width, height), color = (200, 200, 200))
        return error_img, f"Error: {str(e)}"

# --- Create and Launch the Gradio Interface ---
with gr.Blocks() as iface:
    gr.Markdown("# Simple Stable Diffusion UI (Local Model Selection)")
    gr.Markdown(f"Enter a prompt and parameters to generate an image locally. Models in `{CUSTOM_MODEL_DIR}` with extensions {ALLOWED_EXTENSIONS} will be listed below.")

    with gr.Row():
        with gr.Column(scale=2):
            model_selection = gr.Dropdown(choices=available_models, label="Select Model", value="Default (Hugging Face)")
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
        inputs=[prompt_input, negative_prompt_input, steps_input, guidance_input, width_input, height_input, seed_val_input, model_selection],
        outputs=[output_image, seed_output_info]
    )

    gr.Markdown(f"""
    **Notes:**
    * Select a model from the dropdown. Models with extensions {ALLOWED_EXTENSIONS} found in `{CUSTOM_MODEL_DIR}` are listed.
    * "Default (Hugging Face)" will use the standard "{DEFAULT_MODEL_ID}" model.
    * The first image generation after selecting a new model might be slower as it loads into memory.
    * Ensure your `Width` and `Height` are multiples of 64 or 8 for best compatibility.
    * Higher inference steps generally mean better quality but take longer.
    * Guidance scale controls how much the model adheres to your prompt.
    * If running on CPU, expect significantly longer generation times.
    """)

if __name__ == '__main__':
    print("Launching Gradio interface...")
    iface.launch()
    print("Gradio interface should be running. Check your terminal for the local URL (usually http://127.0.0.1:7860 or similar).")
