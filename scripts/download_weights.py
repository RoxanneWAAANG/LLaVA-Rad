import os
from huggingface_hub import snapshot_download
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

def download_llava_rad():
    """
    Download the LLaVA-RAD model weights from Hugging Face Hub
    """
    print("Starting to download LLaVA-RAD model...")
    
    # Create a directory to store the model
    target_path = "/home/jack/Projects/yixin-llm/yixin-llm-data/LLaVA-Rad"
    os.makedirs(target_path, exist_ok=True)
    
    # Download the model snapshot
    model_path = snapshot_download(
        repo_id="microsoft/llava-rad",
        local_dir=target_path,
        token=None  # Add your HF token here if the model is gated
    )
    
    print(f"Model downloaded to: {model_path}")
    return model_path

def load_and_verify_model(model_path):
    """
    Load the model to verify it works correctly
    """
    try:
        # Load the model and processor
        print("Loading the model to verify...")
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("Model loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

if __name__ == "__main__":
    model_path = download_llava_rad()
    success = load_and_verify_model(model_path)