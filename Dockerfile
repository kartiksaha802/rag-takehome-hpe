# Use Python 3.11 slim image as base (Lightweight & Secure)
FROM python:3.11-slim

# Set working directory to /app
WORKDIR /app

# Install system dependencies (Required for llama-cpp-python)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# --- CRITICAL OPTIMIZATION ---
# 1. Install CPU-only PyTorch first.
# This prevents 'sentence-transformers' from pulling the massive GPU version.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the dependencies.
# Pip will see 'torch' is already installed and skip the huge download.
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY data/ ./data/
COPY notebooks/ ./notebooks/
COPY download_model.py .

# Create directories
RUN mkdir -p models chroma_db

# Expose Jupyter port
EXPOSE 8888

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]