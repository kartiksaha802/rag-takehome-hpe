# Minimal RAG: Local CPU Optimization (Llama 3.2 1B)

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Llama 3.2](https://img.shields.io/badge/Model-Llama_3.2_1B-meta?logo=meta)
![Status](https://img.shields.io/badge/Status-POC_Ready-success)
![Docker](https://img.shields.io/badge/Deployment-Docker_Ready-blue?logo=docker)

## 📋 Executive Summary
This repository contains a reference implementation of a **Retrieval-Augmented Generation (RAG)** system optimized for resource-constrained edge environments (Local CPU).

Designed as a "Starter Kit" for enterprise AI adoption, it demonstrates how to achieve low-latency semantic search (<50ms) and generation (~30 tokens/s) without requiring expensive GPU infrastructure. The solution is container-ready and structured to facilitate migration to scalable platforms like **Kubernetes** or **HPE Private Cloud AI**.

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
    git clone [https://github.com/your-username/minimal-rag-local.git](https://github.com/your-username/minimal-rag-local.git)
    cd minimal-rag-local
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
    jupyter notebook notebooks/demo_presentation.ipynb
    ```
    *Note: The first time you run the notebook, it will automatically download the Llama 3.2 model (~800MB).*

---

## 🐳 Deployment (Docker Support)

This project is cloud-native ready. While the notebook is designed for interactive exploration, the environment can be packaged for deployment.

**Build the Image:**
```bash
docker build -t minimal-rag-demo .

graph TD
    %% Styling
    classDef hardware fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef logic fill:#fff3e0,stroke:#ff6f00,stroke-width:2px;
    classDef storage fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    subgraph "HPE Private Cloud AI (Simulated Edge Node)"
        direction TB
        
        User([👤 Customer Engineer]) -->|1. Query| UI[Jupyter Notebook Interface]
        
        subgraph "Application Layer (src/rag_engine.py)"
            UI -->|2. Send Query| Orch{RAG Orchestrator}
            
            subgraph "Ingestion Pipeline"
                Doc[📄 PDF/Text Docs] -->|Chunking| Chunks[Text Chunks]
                Chunks -->|Encode| Embed[Embedder Model<br/>(MiniLM-L6-v2)]
            end
            
            subgraph "Retrieval System"
                Embed -->|3. Vector Search| VDB[(ChromaDB<br/>Persistent Store)]
                VDB -->|4. Return Top-k Chunks| Orch
            end
            
            subgraph "Generation System"
                Orch -->|5. Context + Prompt| LLM[🤖 Inference Engine<br/>Llama-3.2-1B (Int4)]
                LLM -->|6. Generated Answer| Orch
            end
        end
        
        Orch -->|7. Final Response| UI
    end

    %% Apply Styles
    class UI,Orch logic;
    class VDB,Doc hardware;
    class LLM storage;