import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import time
import os

# --- Configuration ---
CUSTOM_MODEL_DIR = "/teamspace/studios/this_studio/SD_models"
ALLOWED_EXTENSIONS = (".safetensors",)
AUTOMATIC_NEGATIVE_EMBEDDINGS = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face" # Example embeddings
# You can also use a list of strings:
# AUTOMATIC_NEGATIVE_EMBEDDINGS = ["ugly", "tiling", "poorly drawn hands", "watermark"]

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
                models.append(item)
    return models

# Get the initial list of available models
available_models = get_available_models(CUSTOM_MODEL_DIR)

# --- Load the Stable Diffusion Pipeline ---
def load_pipeline(model_name):
    model_path = os.path.join(CUSTOM_MODEL_DIR, model_name)
    print(f"Loading model from: {model_path} onto {device} with dtype {torch_dtype}...")
    try:
        if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
            pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch_dtype, use_safetensors=True)
        else:
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        pipe = pipe.to(device)
        if device == "cuda":
            pipe.enable_attention_slicing()
        print("Model loaded successfully.")
        return pipe
    except Exception as e:
        print(f"Error loading the model: {e}")
        print(f"Please ensure the file exists at: {model_path}")
        return None

# Initialize pipeline to None
pipeline = None
initial_model_loaded = False
current_loaded_model_path = None

# --- Define the Image Generation Function for Gradio ---
def generate_image(prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, seed_input, selected_model):
    global pipeline, initial_model_loaded, current_loaded_model_path, AUTOMATIC_NEGATIVE_EMBEDDINGS

    model_path = os.path.join(CUSTOM_MODEL_DIR, selected_model)

    if not initial_model_loaded or current_loaded_model_path != model_path:
        new_pipeline = load_pipeline(selected_model)
        if new_pipeline:
            pipeline = new_pipeline
            initial_model_loaded = True
            current_loaded_model_path = model_path
        else:
            return Image.new('RGB', (width, height), color = (255, 0, 0)), f"Error: Could not load model: {selected_model}"

    if pipeline is None:
        return Image.new('RGB', (width, height), color = (255, 0, 0)), "Error: Stable Diffusion pipeline not loaded."

    # Append automatic negative embeddings
    full_negative_prompt = negative_prompt if negative_prompt else ""
    if isinstance(AUTOMATIC_NEGATIVE_EMBEDDINGS, str):
        if full_negative_prompt:
            full_negative_prompt += ", " + AUTOMATIC_NEGATIVE_EMBEDDINGS
        else:
            full_negative_prompt = AUTOMATIC_NEGATIVE_EMBEDDINGS
    elif isinstance(AUTOMATIC_NEGATIVE_EMBEDDINGS, list):
        if full_negative_prompt:
            full_negative_prompt += ", " + ", ".join(AUTOMATIC_NEGATIVE_EMBEDDINGS)
        else:
            full_negative_prompt = ", ".join(AUTOMATIC_NEGATIVE_EMBEDDINGS)

    print(f"\n--- Generating Image ---")
    print(f"Prompt: '{prompt}'")
    print(f"Negative Prompt: '{full_negative_prompt}'") # Print the full negative prompt
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
                        negative_prompt=full_negative_prompt if full_negative_prompt else None,
                        num_inference_steps=int(num_inference_steps),
                        guidance_scale=float(guidance_scale),
                        width=int(width),
                        height=int(height),
                        generator=generator
                    ).images[0]
            else:
                image = pipeline(
                    prompt,
                        negative_prompt=full_negative_prompt if full_negative_prompt else None,
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
# --- Create and Launch the Gradio Interface ---
with gr.Blocks() as iface:
    gr.Markdown("# Simple Stable Diffusion UI (Local Models Only)")
    gr.Markdown(f"Enter a prompt and parameters to generate an image locally. Available `.safetensors` files in `{CUSTOM_MODEL_DIR}` are listed below. Automatic negative embeddings are applied.")

    with gr.Row():
        with gr.Column(scale=2):
            model_choices = available_models if available_models else ["No models found"]
            initial_value = model_choices[0] if model_choices else None
            model_selection = gr.Dropdown(choices=model_choices, label="Select Model", value=initial_value)
            prompt_input = gr.Textbox(label="Prompt", value="A majestic lion in a lush jungle, photorealistic", lines=3)
            negative_prompt_input = gr.Textbox(label="Negative Prompt (optional)", placeholder="e.g., specific unwanted details", lines=2)
            with gr.Row():
                steps_input = gr.Slider(minimum=10, maximum=150, step=1, value=25, label="Inference Steps")
                guidance_input = gr.Slider(minimum=1.0, maximum=25.0, step=0.1, value=7.5, label="Guidance Scale")
            with gr.Row():
                width_input = gr.Slider(minimum=256, maximum=1024, step=64, value=512, label="Width")
                height_input = gr.Slider(minimum=256, maximum=1024, step=64, value=512, label="Height")
            seed_val_input = gr.Number(label="Seed (-1 or 0 for random)", value=-1, precision=0) # Defined here
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
    * Select a `.safetensors` model file from the dropdown. Ensure you have placed your model files in `{CUSTOM_MODEL_DIR}`.
    * The negative prompt will automatically include: "{AUTOMATIC_NEGATIVE_EMBEDDINGS}" (in addition to anything you type).
    * The first `.safetensors` file found will be selected by default (if any are present).
    * The first image generation after selecting a model might be slower as it loads into memory.
    * Ensure your `Width` and `Height` are multiples of 64 or 8 for best compatibility.
    * Higher inference steps generally mean better quality but take longer.
    * Guidance scale controls how much the model adheres to your prompt.
    * If running on CPU, expect significantly longer generation times.
    """)

if __name__ == '__main__':
    print("Launching Gradio interface...")
    iface.launch()
    print("Gradio interface should be running. Check your terminal for the local URL (usually http://127.0.0.1:7860 or similar).")
