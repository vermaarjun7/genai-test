# Installation Instructions
```
# Create conda environment
conda create -n ml-vector-test python=3.11 -y
conda activate ml-vector-test

# Install required packages
pip install -r requirements.txt
```

# Code Architecture & Components

## System Initialization
- Creates FAQ retrieval system with **TinyLlama (1.1B parameter model)**
- Connects to **Milvus** vector database for similarity search
- Loads **Sentence Transformer** for embedding generation
- Optimizes model for CPU inference

---

## Data Flow Process
**Pipeline:**
```
Question Input → Embedding Generation → Vector Search → Context Retrieval → Answer Generation
```
- Uses **cosine similarity** to find most relevant FAQ entries  
- Routes **high-similarity matches (>0.85)** directly without LLM processing  
- Falls back to **rule-based responses** if model fails  


## Core Components

### Embedding Model
- **Model:** `all-MiniLM-L6-v2` (384-dimensional vectors)  
- Converts **text queries** and **FAQ entries** into numerical representations  
- Enables **semantic similarity search** in vector space  

### TinyLlama Model
- **1.1B parameter language model** optimized for speed  
- Runs in **float32** for CPU compatibility  
- Inference optimizations:
  - `eval()` mode  
  - Gradient disabling  
- Uses **chat template formatting** for better responses  

### Vector Database (Milvus)
- Stores **FAQ embeddings** with cosine similarity index  
- Performs **fast approximate nearest neighbor search**  
- Returns **top-k most relevant FAQ entries** with similarity scores  

### Smart Routing Logic
- **Score > 0.9:** Direct FAQ response (instant)  
- **Score < 0.9:** LLM enhancement (~1–2 seconds)  
- **LLM Fails:** Rule-based or fallback response  



