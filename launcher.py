import subprocess
import sys
import os
import platform

# --- Configuration ---
UI_SCRIPT_NAME = "simple_sd_ui.py"

# Base requirements (Pillow is for PIL)
REQUIREMENTS = [
    "diffusers",
    "transformers",
    "accelerate",
    "gradio",
    "Pillow"
]

# PyTorch installation commands
# For CPU or as a general fallback
TORCH_CPU_COMMAND = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
# For NVIDIA CUDA 11.8 (a common recent version)
TORCH_CUDA_COMMAND = [sys.executable, "-m", "pip", "install", "torch==2.1.2", "torchvision==0.16.2", "torchaudio==2.1.2", "--index-url", "https://download.pytorch.org/whl/cu118"]
# For Apple Silicon (M1/M2 Macs) - will typically grab the correct version
TORCH_MPS_COMMAND = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]


def check_if_ui_script_exists():
    """Checks if the main UI script is present in the current directory."""
    if not os.path.exists(UI_SCRIPT_NAME):
        print(f"Error: The UI script '{UI_SCRIPT_NAME}' was not found in the current directory.")
        print(f"Please make sure '{UI_SCRIPT_NAME}' (the Gradio UI code) is in the same folder as this launcher.")
        sys.exit(1)

def install_package(package_name=None, command=None, package_display_name=None):
    """Installs a package using pip, either by name or by a full command list."""
    install_name = package_display_name or package_name or " ".join(command)
    try:
        if command:
            print(f"Running installation: {' '.join(command)}")
            subprocess.check_call(command)
        elif package_name:
            print(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully processed installation for {install_name}.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {install_name}: {e}\n")
        return False
    except FileNotFoundError:
        print("Error: 'pip' command not found. Make sure Python and pip are installed and in your system's PATH.")
        sys.exit(1)

def install_pytorch():
    """Attempts to install PyTorch, guiding the user if possible."""
    print("\n--- Step 1: Installing PyTorch ---")
    print("This script will attempt to install PyTorch.")
    print("For the best performance, especially with NVIDIA GPUs, you might need to install PyTorch manually")
    print("with the correct CUDA version from: https://pytorch.org/get-started/locally/")
    print("-" * 50)

    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    is_windows = platform.system() == "Windows"
    is_linux = platform.system() == "Linux"

    if is_apple_silicon:
        print("Apple Silicon (M1/M2) detected. Attempting to install PyTorch with MPS support.")
        if install_package(command=TORCH_MPS_COMMAND, package_display_name="PyTorch for Apple Silicon"):
            return True
        else:
            print("PyTorch installation for Apple Silicon failed. Please install it manually from pytorch.org.")
            return False
    else: # Windows or Linux (or other)
        print("For NVIDIA GPUs, PyTorch needs to be compiled with the correct CUDA version.")
        print("We will attempt to install a version for CUDA 11.8.")
        print("If you don't have an NVIDIA GPU, or have a different CUDA version, this might not be optimal,")
        print("and the UI script will likely fall back to CPU mode (which is very slow).")

        use_cuda_default = input("Do you have an NVIDIA GPU and want to try installing PyTorch with CUDA 11.8 support? (Y/n): ").strip().lower()

        if use_cuda_default in ['', 'y', 'yes']:
            print("\nAttempting to install PyTorch with CUDA 11.8 support...")
            if install_package(command=TORCH_CUDA_COMMAND, package_display_name="PyTorch for CUDA 11.8"):
                # A basic check if torch can be imported and CUDA is seen
                try:
                    import torch
                    if torch.cuda.is_available():
                        print("PyTorch with CUDA support seems to be installed and CUDA is available to PyTorch.")
                    else:
                        print("PyTorch (CUDA version) installed, but torch.cuda.is_available() is False.")
                        print("This could mean an issue with your NVIDIA drivers, CUDA toolkit installation, or the PyTorch build.")
                        print("The UI script will likely try to use the CPU. For GPU acceleration, ensure your NVIDIA setup is correct and consider a manual PyTorch install.")
                except ImportError:
                    print("Could not import torch after CUDA installation attempt. This is unexpected.")
                except Exception as e:
                    print(f"Error checking torch after installation: {e}")
                return True # Proceed even if CUDA check has warnings, UI script will adapt
            else:
                print("PyTorch with CUDA installation failed.")
                # Fall through to CPU attempt
        else:
            print("Skipping CUDA-specific PyTorch installation based on user input.")

        print("\nAttempting to install a general/CPU-compatible version of PyTorch.")
        if install_package(command=TORCH_CPU_COMMAND, package_display_name="PyTorch (CPU/General)"):
            print("PyTorch (CPU/General version) installation command executed.")
            print("If you have an NVIDIA GPU, for best performance, please ensure you have the correct CUDA-enabled PyTorch build installed from pytorch.org.")
            return True
        else:
            print("PyTorch CPU/General installation also failed. Please install PyTorch manually from https://pytorch.org/get-started/locally/ before proceeding.")
            return False

def main():
    """Main function to install requirements and launch the UI."""
    print("--- Stable Diffusion UI Auto-Installer & Launcher ---")
    print("This script will attempt to install required Python packages and then launch the UI.")
    print("=" * 60)

    check_if_ui_script_exists()

    if not install_pytorch():
        print("\nPyTorch installation failed or was not fully successful. The UI might not run correctly or use the CPU.")
        print("Please ensure PyTorch is correctly installed for your system (see https://pytorch.org/) and try again.")
        sys.exit(1)
    print("\nPyTorch installation process completed.")

    print("\n--- Step 2: Installing other requirements ---")
    all_other_reqs_installed = True
    for req in REQUIREMENTS:
        if not install_package(package_name=req):
            all_other_reqs_installed = False
            # No need to break, let it try all, then report.
    print("-" * 50)

    if not all_other_reqs_installed:
        print("One or more standard requirements could not be installed. Please check the errors above.")
        print(f"Ensure you can install packages with '{sys.executable} -m pip install <package_name>'")
        sys.exit(1)

    print("\nAll requirements should now be processed.")
    print(f"\n--- Step 3: Launching {UI_SCRIPT_NAME} ---")
    print("=" * 60)

    try:
        # Run the UI script using the same Python interpreter
        # This ensures it uses the environment where packages were just installed.
        subprocess.run([sys.executable, UI_SCRIPT_NAME], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {UI_SCRIPT_NAME}: {e}")
        print("The UI script might have encountered an error during its execution.")
    except FileNotFoundError:
        # This should have been caught by check_if_ui_script_exists or sys.executable issue
        print(f"Error: Could not find Python interpreter '{sys.executable}' or the script '{UI_SCRIPT_NAME}'.")
    except Exception as e:
        print(f"An unexpected error occurred while trying to launch {UI_SCRIPT_NAME}: {e}")

if __name__ == "__main__":
    main()
