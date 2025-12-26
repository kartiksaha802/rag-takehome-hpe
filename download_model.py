import os
from huggingface_hub import hf_hub_download

# Constants
REPO_ID = "bartowski/Llama-3.2-1B-Instruct-GGUF"
FILENAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
MODEL_DIR = "./models"

print(f"⬇️  Starting download from {REPO_ID}...")
print(f"📂 Saving to {MODEL_DIR}...")

# This function handles the download and caching automatically
file_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False  # Crucial: This ensures you get the actual file, not a shortcut
)

print(f"✅ Success! Model saved at: {file_path}")