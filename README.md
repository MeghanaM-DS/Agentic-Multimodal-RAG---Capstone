# Agentic-Multimodal-RAG---Capstone

# 🏥 Medical Imaging RAG System (CT/MRI Multimodal Retrieval + LLM Summarization)

A research-focused Retrieval-Augmented Generation (RAG) system for CT and MRI medical imaging, combining:

- FAISS-based medical image similarity search
- Multimodal retrieval (text + image)
- Hybrid embeddings (image + text concatenation)
- LLM reasoning over retrieved clinical cases
- Support for OpenRouter, OpenAI, and local Ollama models
- A clean Streamlit clinical-style UI
- A FastAPI backend serving retrieval + RAG endpoints

---

## ⚠️ Disclaimer

This system is for **research and educational use only**.  
It is **NOT** a diagnostic tool and must **NOT** be relied upon for clinical decision-making.

---

## 📘 Overview

This project helps clinicians, researchers, and students:

- Upload CT/MRI images
- Enter a clinical question
- Retrieve similar reference cases from a FAISS index
- Feed retrieved cases into an LLM
- Generate a **safe, pattern-based summary** of what the retrieved cases show

### 🔍 Example use cases

- “What patterns appear in cases similar to this uploaded CT image?”
- “Summarize typical findings for ovarian torsion on CT based on retrieved examples.”
- “Given this CT and my query, what do the closest MRI cases look like?”

The system **never** generates diagnoses.  
It only summarizes **patterns observed in retrieved cases**.

---

## 🧠 High-Level Architecture

```text
User (Streamlit UI)
        ↓
FastAPI Backend
        ↓
| Retrieval (FAISS Index) |
| - Text search           |
| - Image search          |
| - Multimodal search     |
        ↓
Retrieved CT/MRI Metadata
        ↓
| RAG Pipeline            |
| LLM Router (OpenRouter, |
| OpenAI, Ollama)         |
        ↓
LLM Summary (non-diagnostic)
        ↓
Displayed in UI
```

---

## 📁 Project Structure

```text
rag/
├── app.py # FastAPI backend
├── streamlit_app.py # Clinical-style UI
│
├── src/
│ ├── components/
│ │ ├── data_loader.py # Dataset loading utilities
│ │ ├── embeddings.py # Text + image embedding models
│ │ ├── indexing.py # FAISS index class (load/save)
│ │ ├── retrieval.py # Main multimodal retrieval system
│ │
│ ├── llm/
│ │ ├── base.py # LLM config + enums
│ │ ├── router.py # OpenRouter / OpenAI / Ollama abstractions
│ │
│ ├── services/
│ ├── rag_service.py # Orchestrates retrieval + LLM generation
│
├── models/
│ └── faiss_indices/ # Prebuilt FAISS hybrid index + metadata
│
├── data/
│ ├── raw/ # Raw dataset images/metadata
│ └── processed/ # Cleaned embeddings & structured data
│
├── requirements.txt
└── README.md
```

---

## 🔧 Prerequisites

### 1. Python 3.9+

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Required FAISS Index

Your directory must contain:

```pgsql
models/faiss_indices/
   ├── hybrid_concat.index
   └── hybrid_concat_metadata.json
```

If missing → rebuild index using your dataset.

### 4. API Keys (optional depending on provider)

#### OpenRouter

```bash
export OPENROUTER_API_KEY="your_key_here"
```

#### OpenAI

```bash
export OPENAI_API_KEY="your_key_here"
```

#### Ollama (local)

Install: https://ollama.ai/download

#### Run server:

```bash
ollama serve
```

#### Install Gemma:

```bash
ollama pull gemma3:4b
```

## 🚀 Running the Backend (FastAPI)

```bash
uvicorn app:app --reload
```

API Docs:
http://127.0.0.1:8000/docs

## 🖼️ Running the UI (Streamlit)

```bash
streamlit run streamlit_app.py
```

Open in browser: http://localhost:8501

You will see:

- Sidebar model controls
- Query input
- Optional CT/MRI image upload
- AI summary (“Impression”)
- Retrieved context
- Raw FAISS results

## 📡 API Endpoints (Summary)

### Health Check

```bash
GET /health
GET /stats
```

### Retrieval

```pgsql
POST /search/text
POST /search/image
```

### RAG (main feature)

```bash
POST /rag/query         # text-only RAG
POST /rag/multimodal    # image + text RAG
```

Each returns:

- answer → LLM-generated pattern summary
- retrieval_context → text passed into LLM
- raw_results → top-k FAISS results
- query and optional image_path
- query_id → unique identifier for metrics tracking

### Metrics Endpoints

```bash
GET  /metrics/summary        # Performance summary
GET  /metrics/aggregated     # Detailed aggregated metrics
GET  /metrics/recent?n=10    # Last N queries
POST /metrics/save           # Save metrics to disk
POST /metrics/reset          # Clear all metrics
```

## 🔍 How Retrieval Works

1. Image embeddings

   - CLIP-like encoder
   - ~512 dimensions

2. Text embeddings

   - SentenceTransformer MiniLM
   - ~384 dimensions

3. Hybrid embedding

```python
final_vector = concat(image_vector, text_vector)
```

This allows:

- Text-only queries
- Image-only queries
- Multimodal (text + image) queries

All inside one unified FAISS index.

## 🤖 How RAG Works

#### Step 1 - Retrieve

FAISS returns metadata for similar CT/MRI cases.

#### Step 2 - Format Retrieval Context

`retriever.format_results_for_display()`

Creates structured summaries like:

- Diagnosis
- Modality
- Findings
- Body region

#### Step 3 - LLM Router

Selects backend:

- OpenRouter → GPT-4o, Claude, Llama, etc.
- OpenAI → GPT-4.1, GPT-4o-mini
- Ollama → local Gemma, Llama

#### Step 4 - Safety Prompting

Ensures:

- NO diagnoses
- Only patterns from retrieved cases
- Clinical caution

#### Step 5 - Generate Answer

- Returned to Streamlit UI.

## 🧪 API Testing Examples

### Text Retrieval

```json
POST /search/text
{
  "query_text": "brain hemorrhage",
  "top_k": 3
}
```

### Image Retrieval

Upload a CT/MRI image → /search/image

### Text RAG

```json
POST /rag/query
{
  "query_text": "What patterns are seen in ovarian torsion?",
  "top_k": 3,
  "llm_provider": "openrouter",
  "llm_model": "openai/gpt-4o-mini"
}
```

### Multimodal RAG

Upload an image and pass a query.

## 🎨 Streamlit UI Features

### RAG Query Tab

- Clinical-themed layout
- Sidebar with:
  - Provider selector (OpenRouter / OpenAI / Ollama)
  - Model dropdown
  - Temperature control
  - Max tokens
  - Top-k sliders
- Main interface:
  - Query text input
  - Optional image upload
  - Image preview
  - AI impression summary
  - Retrieved clinical context
  - Raw FAISS results
  - Similarity bar visualization

### Metrics Tab

- **Key Performance Indicators (KPIs)**:
  - Total queries processed
  - Success rate percentage
  - Average response time
  - Average similarity scores
- **Performance Breakdown**:
  - Retrieval time vs LLM time
  - Query type distribution (text/image/multimodal)
- **LLM Usage Statistics**:
  - Provider distribution
  - Model usage tracking
  - Total tokens consumed
- **Recent Query History**:
  - Timestamp, query type, timings
  - Success/failure status
  - Similarity scores
- **Error Tracking**:
  - Failed query count
  - Error type breakdown
- **Controls**:
  - Refresh metrics
  - Save summary to disk
  - Export logs

## 📊 Metrics & Monitoring

The system includes comprehensive metrics tracking for:

### Tracked Metrics

1. **Query Metrics**
   - Query type (text/image/multimodal)
   - Timestamp and unique ID
   - Input parameters (top-k, modality, etc.)

2. **Performance Metrics**
   - Retrieval time (ms)
   - LLM generation time (ms)
   - Total end-to-end time
   - Number of results returned

3. **Quality Metrics**
   - Average similarity scores
   - Max/min similarity in results
   - Success/failure rates

4. **LLM Usage**
   - Provider and model used
   - Token consumption
   - Error rates and types

### Viewing Metrics

**Option 1: Streamlit UI**
- Navigate to the "Metrics" tab
- View real-time KPIs and charts
- Export summaries

**Option 2: Standalone Dashboard**
```bash
streamlit run metrics_dashboard.py
```

**Option 3: API Endpoints**
```bash
curl http://127.0.0.1:8000/metrics/summary
```

### Metrics Storage

- In-memory tracking for session
- Automatic logging to `logs/metrics/`
- Daily JSONL logs: `metrics_YYYYMMDD.jsonl`
- Summary exports: `summary_YYYYMMDD_HHMMSS.json`

---

## 📊 Evaluation System

### Comprehensive Metrics

The system includes enterprise-grade evaluation capabilities with **15+ metrics**:

**RAGAS Metrics** (RAG-specific evaluation):
- **Faithfulness**: Is the answer grounded in retrieved contexts?
- **Answer Relevancy**: Does the answer address the query?
- **Context Relevancy**: Are retrieved contexts relevant?
- **Context Recall**: Is ground truth info in contexts?
- **Context Precision**: Are contexts precise and focused?

**Traditional NLG Metrics**:
- **F1 Score**: Token overlap accuracy
- **BLEU**: N-gram precision
- **ROUGE-1, ROUGE-2, ROUGE-L**: Various overlap measures

**Retrieval Metrics**:
- **Precision@K & Recall@K**: Retrieval accuracy
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Ranking quality
- **Average Similarity**: Cosine similarity scores

### Using the Evaluation System

**1. Evaluation Dashboard**
```bash
streamlit run evaluation_dashboard.py
```
- Interactive UI with radar charts, bar charts, gauge charts
- Submit queries for evaluation
- View RAGAS, F1, BLEU, ROUGE metrics
- Save/export evaluation reports

**2. API Endpoints**
```python
# Evaluate a query
POST /evaluate/query
{
  "query": "What is shown in this CT scan?",
  "generated_answer": "The CT shows...",
  "retrieved_contexts": ["Context 1", "Context 2"],
  "ground_truth": "The CT scan reveals..."
}

# Get aggregated metrics
GET /evaluate/aggregated

# Save evaluation report
POST /evaluate/save
```

**3. Programmatic Usage**
```python
from src.evaluation.rag_evaluation import RAGEvaluator

evaluator = RAGEvaluator(save_dir=Path("logs/evaluation"))

result = evaluator.evaluate(
    query="What abnormalities are visible?",
    generated_answer="The scan shows...",
    retrieved_contexts=["Context 1", "Context 2"],
    ground_truth="Ground truth answer"
)

print(f"Faithfulness: {result.faithfulness:.4f}")
print(f"F1 Score: {result.f1_score:.4f}")
print(f"ROUGE-1: {result.rouge_1:.4f}")
```

**4. Testing**
```bash
python test_evaluation.py
```

### Documentation

- **Quick Start**: See `EVALUATION_QUICKSTART.md`
- **Complete Guide**: See `docs/EVALUATION_GUIDE.md`
- **Integration Example**: See `example_evaluation_integration.py`

### Evaluation Files

```
src/evaluation/
└── rag_evaluation.py          # Core evaluation module

evaluation_dashboard.py         # Streamlit dashboard
test_evaluation.py              # Test script

logs/evaluation/                # Evaluation logs
├── evaluation_YYYYMMDD.jsonl   # Daily logs
└── eval_report_*.json          # Saved reports
```

---

## 🛠️ Future Extensions

- Add X-ray modality
- Cross-modal retrieval (CT ↔ MRI ↔ X-ray)
- Display retrieved image thumbnails
- Radiology lexicon grounding
- Better safety filtering
- CFR / HIPAA-style disclaimers
- LLM-based evaluation (GPT-4 as judge)
- Semantic similarity metrics (BERTScore)
- A/B testing framework for model comparison
- Prometheus/Grafana integration

## 🏁 Final Notes

This project provides a powerful platform for:

- Radiology education
- Case-based medical retrieval
- Multimodal RAG experimentation
- Safe pattern summarization using LLMs
- **Comprehensive evaluation with RAGAS, F1, BLEU, ROUGE metrics**

## Data Used So Far:

- MedPix 2.0 : https://github.com/CHILab1/MedPix-2.0/tree/main/MedPix-2-0