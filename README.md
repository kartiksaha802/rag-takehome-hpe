# Minimal RAG: Local CPU Optimization (Llama 3.2 1B)

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Llama 3.2](https://img.shields.io/badge/Model-Llama_3.2_1B-meta?logo=meta)
![Docker](https://img.shields.io/badge/Deployment-Docker_Ready-blue?logo=docker)

## 📋 Executive Summary
This repository contains a reference implementation of a **Retrieval-Augmented Generation (RAG)** system optimized for resource-constrained edge environments (Local CPU).

Designed as a "Starter Kit" for enterprise AI adoption, it demonstrates how to achieve low-latency semantic search (<50ms) and generation (~30 tokens/s) without requiring expensive GPU infrastructure. The solution is container-ready.

---

## 🏗️ System Architecture

The system is architected to balance **simplicity** (for rapid prototyping) with **extensibility** (for production migration).

| Component | Current Implementation (Local) | Production Target (Enterprise) |
| :--- | :--- | :--- |
| **Inference Engine** | `Llama-3.2-1B-Instruct` (Int4 Quantized via `llama.cpp`) | **NVIDIA NIM** or **Triton Inference Server** (GPU) |
| **Vector Store** | **ChromaDB** (Persistent Local Client) | **Milvus** or **Weaviate** (Distributed Cluster) |
| **Embeddings** | `all-MiniLM-L6-v2` (384-dim, CPU Optimized) | `bge-m3` or `text-embedding-3-large` |
| **Orchestration** | Modular Python Pipeline | **Kubeflow** or **Airflow** DAGs |

---

## ⚡ Key Features

* **Zero-GPU Requirement:** Runs efficiently on standard laptop CPUs using GGUF quantization (Int4).
* **Self-Healing Setup:** The system automatically checks for and downloads missing model artifacts from HuggingFace upon initialization.
* **Integrated Telemetry:** Real-time observability into Retrieval Latency, Cosine Similarity Scores, and Generation Time to detect hallucinations.
* **Clean Architecture:** Separation of concerns between the Core Logic (`src/`) and Presentation Layer (`notebooks/`).
* **Container Ready:** Includes a `Dockerfile` for reproducible deployment in containerized environments.

---

## 🚀 Getting Started

### Prerequisites
* **Python 3.10+** installed.
* **RAM:** Minimum 4GB available (8GB recommended).
* **Disk Space:** ~2GB (for Model + Libraries).

### Installation (Local)

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/kartiksaha802/rag-takehome-hpe.git
    cd rag-takehome-hpe
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Demo**
    Launch the Jupyter Notebook to explore the solution interactively.
    ```bash
    jupyter notebook notebooks/demo.ipynb
    ```
    *Note: The first time you run the notebook, it will automatically download the Llama 3.2 model (~800MB).*

---

## 🐳 Deployment (Docker Support)

This project is cloud-native ready. While the notebook is designed for interactive exploration, the environment can be packaged for deployment.

**Build the Image:**
```bash
docker build -t minimal-rag-demo .

