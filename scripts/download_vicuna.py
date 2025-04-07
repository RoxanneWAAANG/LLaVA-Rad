import os
from huggingface_hub import snapshot_download

# Set your Hugging Face token if needed
# os.environ["HF_TOKEN"] = "your_token_here"  # Uncomment and add token if the model requires authentication

# Directory to save the model
output_dir = "/home/jack/Projects/yixin-llm/yixin-llm-data/LLaVA-Rad/vicuna-7b-v1.5"

# Download the model
print(f"Downloading lmsys/vicuna-7b-v1.5 to {output_dir}...")
snapshot_download(
    repo_id="lmsys/vicuna-7b-v1.5",
    local_dir=output_dir,
    local_dir_use_symlinks=False  # Set to True to save disk space with symlinks
)

print(f"Download complete! Model saved to {output_dir}")