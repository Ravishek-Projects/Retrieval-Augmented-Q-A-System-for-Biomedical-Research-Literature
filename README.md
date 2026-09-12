# 🏥 Medical Literature RAG System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medical-literature-rag.streamlit.app/)

**Author:** Ravishek Kumar

A Retrieval-Augmented Generation (RAG) system built to answer complex medical and biomedical questions. The system grounds its answers in a curated dataset of over **100,000 arXiv medical research papers**, ensuring highly factual, hallucination-free responses with exact source citations.

🔗 **[Live Demo](https://medical-literature-rag.streamlit.app/)**

---

## ✨ Features
* **Fact-Grounded Answers:** Uses Groq's fast LLM (`gpt-oss-120b`) to synthesize answers exclusively from retrieved abstracts.
* **Inline Citations:** The model strictly cites its sources, linking directly back to the original arXiv papers.
* **Hallucination Guard:** Automatically refuses to answer if the provided context lacks sufficient information.
* **Two-Stage Retrieval Pipeline:** Uses a lightweight Bi-encoder for fast vector search, followed by an advanced Cross-encoder for high-precision re-ranking.
* **Lightning Fast:** Fully optimized to run on CPU-only infrastructure without sacrificing retrieval accuracy.

---

## 🏗️ Architecture

The system uses a robust two-stage retrieval pipeline:

```mermaid
flowchart LR
    Q[User Question] --> B[Bi-Encoder]
    B -->|all-MiniLM-L6-v2| V[(FAISS Index\n100k Vectors)]
    V -->|Top 50 Candidates| C[Cross-Encoder Re-ranker]
    C -->|ms-marco-MiniLM| T[Top 5 Papers]
    T --> L[Groq LLM]
    L -->|gpt-oss-120b| A[Grounded Answer\nwith Citations]
    
    style Q fill:#e1f5fe,stroke:#01579b
    style V fill:#fff3e0,stroke:#e65100
    style A fill:#e8f5e9,stroke:#1b5e20
```

---

## 📊 Evaluation & Metrics

The system was evaluated against an auto-generated dataset of 50 ground-truth Q&A pairs extracted from the medical corpus. 

### Ablation Study (The Impact of Re-ranking)
Implementing the Cross-Encoder re-ranker dramatically improved the system's ability to pull the exact right paper into the top 5 results.

| Configuration | Recall@5 | Recall@10 | Mean Reciprocal Rank (MRR) |
|---|---|---|---|
| No re-ranker (Top 50) | 0.440 | 0.520 | 0.368 |
| **With re-ranker (Top 50)** | **0.760** | **0.780** | **0.743** |

*Adding the re-ranker yielded a **73% relative improvement** in Recall@5.*

### Answer Quality
Using **BERTScore** to measure semantic similarity between the generated answers and the reference answers:
* **F1 Score:** `0.8375`

---

## 🛠️ Tech Stack
* **UI & Hosting:** Streamlit, Streamlit Community Cloud
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Re-ranking:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
* **LLM Provider:** Groq API (`openai/gpt-oss-120b`)
* **Data Processing:** Pandas, PyArrow

---

## 💻 Local Installation

If you want to run this project locally on your machine:

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ravishek-Projects/Retrieval-Augmented-Q-A-System-for-Biomedical-Research-Literature.git
   cd Retrieval-Augmented-Q-A-System-for-Biomedical-Research-Literature
   ```
   *(Note: This repository uses Git LFS for the large index files. Make sure you have Git LFS installed before cloning).*

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Groq API Key**
   * On Windows (Command Prompt): `set GROQ_API_KEY=your_api_key`
   * On Mac/Linux: `export GROQ_API_KEY=your_api_key`

4. **Run the app**
   ```bash
   streamlit run app.py
   ```
