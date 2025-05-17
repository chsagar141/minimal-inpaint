import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import time
import os
import numpy as np

# --- Configuration ---
CUSTOM_MODEL_DIR = "/teamspace/studios/this_studio/SD_models"
ALLOWED_EXTENSIONS = (".safetensors",)
AUTOMATIC_NEGATIVE_EMBEDDINGS = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face" # Example embeddings

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

# --- Load the Stable Diffusion Inpaint Pipeline ---
def load_inpaint_pipeline(model_name):
    model_path = os.path.join(CUSTOM_MODEL_DIR, model_name)
    print(f"Loading inpaint model from: {model_path} onto {device} with dtype {torch_dtype}...")
    try:
        if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
            # Assuming the .safetensors contains necessary components for inpainting
            pipe = StableDiffusionInpaintPipeline.from_single_file(model_path, torch_dtype=torch_dtype, use_safetensors=True)
        else:
            raise FileNotFoundError(f"Inpaint model file not found at: {model_path}")

        pipe = pipe.to(device)
        if device == "cuda":
            pipe.enable_attention_slicing()
        print("Inpaint model loaded successfully.")
        return pipe
    except Exception as e:
        print(f"Error loading the inpaint model: {e}")
        print(f"Please ensure the file exists at: {model_path} and contains inpaint components.")
        return None

# Initialize pipelines
pipeline = None
inpaint_pipeline = None
initial_model_loaded = False
current_loaded_model_path = None
initial_inpaint_model_loaded = False
current_loaded_inpaint_model_path = None

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

    full_negative_prompt = negative_prompt if negative_prompt else ""
    if isinstance(AUTOMATIC_NEGATIVE_EMBEDDINGS, str):
        full_negative_prompt += ", " + AUTOMATIC_NEGATIVE_EMBEDDINGS
    elif isinstance(AUTOMATIC_NEGATIVE_EMBEDDINGS, list):
        full_negative_prompt += ", " + ", ".join(AUTOMATIC_NEGATIVE_EMBEDDINGS)

    print(f"\n--- Generating Image ---")
    print(f"Prompt: '{prompt}'")
    print(f"Negative Prompt: '{full_negative_prompt}'")
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

# --- Define the Inpaint Function for Gradio ---
def inpaint_image(init_image, mask_image, prompt, negative_prompt, num_inference_steps, guidance_scale, seed_input, selected_inpaint_model):
    global inpaint_pipeline, initial_inpaint_model_loaded, current_loaded_inpaint_model_path, AUTOMATIC_NEGATIVE_EMBEDDINGS

    if init_image is None or mask_image is None:
        return Image.new('RGB', (512, 512), color = (255, 0, 0)), "Error: Please upload an initial image and a mask."

    model_path = os.path.join(CUSTOM_MODEL_DIR, selected_inpaint_model)

    if not initial_inpaint_model_loaded or current_loaded_inpaint_model_path != model_path:
        new_inpaint_pipeline = load_inpaint_pipeline(selected_inpaint_model)
        if new_inpaint_pipeline:
            inpaint_pipeline = new_inpaint_pipeline
            initial_inpaint_model_loaded = True
            current_loaded_inpaint_model_path = model_path
        else:
            return Image.new('RGB', (512, 512), color = (255, 0, 0)), f"Error: Could not load inpaint model: {selected_inpaint_model}"

    if inpaint_pipeline is None:
        return Image.new('RGB', (512, 512), color = (255, 0, 0)), "Error: Inpaint pipeline not loaded."

    full_negative_prompt = negative_prompt if negative_prompt else ""
    if isinstance(AUTOMATIC_NEGATIVE_EMBEDDINGS, str):
        full_negative_prompt += ", " + AUTOMATIC_NEGATIVE_EMBEDDINGS
    elif isinstance(AUTOMATIC_NEGATIVE_EMBEDDINGS, list):
        full_negative_prompt += ", " + ", ".join(AUTOMATIC_NEGATIVE_EMBEDDINGS)

    print(f"\n--- Inpainting Image ---")
    print(f"Prompt: '{prompt}'")
    print(f"Negative Prompt: '{full_negative_prompt}'")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    print(f"Seed: {seed_input}")
    print(f"Selected Inpaint Model: '{selected_inpaint_model}'")

    if seed_input == -1 or seed_input == 0:
        seed = int(time.time() * 1000) % (2**32 -1)
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"Using random seed: {seed}")
    else:
        generator = torch.Generator(device=device).manual_seed(int(seed_input))
        print(f"Using fixed seed: {int(seed_input)}")

    try:
        init_image = init_image.convert("RGB").resize((512, 512))
        mask_image = mask_image.convert("L").resize((512, 512))

        with torch.no_grad():
            if device == "cuda" or device == "mps":
                with torch.autocast(device_type=device):
                    inpainted_image = inpaint_pipeline(
                        prompt=prompt,
                        image=init_image,
                        mask_image=mask_image,
                        negative_prompt=full_negative_prompt if full_negative_prompt else None,
                        num_inference_steps=int(num_inference_steps),
                        guidance_scale=float(guidance_scale),
                        generator=generator
                    ).images[0]
            else:
                inpainted_image = inpaint_pipeline(
                        prompt=prompt,
                        image=init_image,
                        mask_image=mask_image,
                        negative_prompt=full_negative_prompt if full_negative_prompt else None,
                        num_inference_steps=int(num_inference_steps),
                        guidance_scale=float(guidance_scale),
                        generator=generator
                    ).images[0]
        print("Image inpainted successfully.")
        return inpainted_image, f"Seed used: {generator.initial_seed() if generator else 'N/A'}"
    except Exception as e:
        print(f"Error during inpainting: {e}")
        error_img = Image.new('RGB', (512, 512), color = (200, 200, 200))
        return error_img, f"Error: {str(e)}"

# --- Create and Launch the Gradio Interface ---

# --- Create and Launch the Gradio Interface ---
with gr.Blocks() as iface:
    gr.Markdown("# Simple Stable Diffusion UI (Generation & Inpaint)")

    with gr.Tab("A. Generation"):
        gr.Markdown("## Image Generation")
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
        * The negative prompt will automatically include: "{AUTOMATIC_NEGATIVE_EMBEDDINGS}" (in addition to anything you type).
        """)
    with gr.Tab("B. Inpaint"):
        gr.Markdown("## Image Inpainting")
        gr.Markdown("Please upload an initial image and a separate mask image (white area indicates where to inpaint, black area remains unchanged).")
        with gr.Row():
            with gr.Column(scale=2):
                inpaint_model_choices = available_models if available_models else ["No models found"]
                initial_inpaint_value = inpaint_model_choices[0] if inpaint_model_choices else None
                inpaint_model_selection = gr.Dropdown(choices=inpaint_model_choices, label="Select Inpaint Model", value=initial_inpaint_value)
                inpaint_init_image = gr.Image(label="Initial Image", type="pil")
                inpaint_mask_image = gr.Image(label="Mask Image (white on black)", type="pil")
                inpaint_prompt_input = gr.Textbox(label="Inpaint Prompt", value="A futuristic robot arm", lines=3)
                inpaint_negative_prompt_input = gr.Textbox(label="Negative Prompt (optional)", placeholder="e.g., unwanted artifacts", lines=2)
                with gr.Row():
                    inpaint_steps_input = gr.Slider(minimum=10, maximum=150, step=1, value=50, label="Inpaint Steps")
                    inpaint_guidance_input = gr.Slider(minimum=1.0, maximum=25.0, step=0.1, value=7.5, label="Inpaint Guidance Scale")
                inpaint_seed_val_input = gr.Number(label="Seed (-1 or 0 for random)", value=-1, precision=0)
                inpaint_button = gr.Button("Inpaint Image", variant="primary")

            with gr.Column(scale=1):
                inpaint_output_image = gr.Image(label="Inpainted Image", type="pil")
                inpaint_seed_output_info = gr.Textbox(label="Inpaint Info", interactive=False)

        inpaint_button.click(
            fn=inpaint_image,
            inputs=[inpaint_init_image, inpaint_mask_image, inpaint_prompt_input, inpaint_negative_prompt_input, inpaint_steps_input, inpaint_guidance_input, inpaint_seed_val_input, inpaint_model_selection],
            outputs=[inpaint_output_image, inpaint_seed_output_info]
        )
        gr.Markdown("""
        **Notes for Inpainting:**
        * Upload an initial image and a separate mask image (white area indicates where to inpaint, black area remains unchanged).
        * Ensure you have a model capable of inpainting selected. Some standard SD models might work, but dedicated inpaint models often yield better results.
        * The negative prompt will also include the standard negative embeddings.
        """)


    gr.Markdown("""
    **General Notes:**
    * Ensure your `Width` and `Height` are multiples of 64 or 8 for best compatibility.
    * Higher inference steps generally mean better quality but take longer.
    * Guidance scale controls how much the model adheres to your prompt.
    * If running on CPU, expect significantly longer generation times.
    """)

if __name__ == '__main__':
    print("Launching Gradio interface...")
    iface.launch()
    print("Gradio interface should be running. Check your terminal for the local URL (usually http://127.0.0.1:7860 or similar).")
