import os
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="Medical Research Q&A",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths (relative to repo root on HF Spaces) ─
INDEX_PATH    = "index/faiss_index.bin"
METADATA_PATH = "index/chunks_metadata.parquet"
GROQ_MODEL    = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a medical research assistant with expertise in biomedical literature.
Answer questions based ONLY on the research paper abstracts provided.
Rules:
1. Ground every claim in the provided context. Do NOT use outside knowledge.
2. Cite sources inline using [1], [2], etc. after each relevant sentence.
3. If the answer is not in context, say exactly: "The provided papers do not contain enough information to answer this question."
4. Keep answers concise and factual — 3 to 5 sentences maximum.
5. End with a Sources: section listing cited papers."""


# ── Load all resources once (cached) ──────────
@st.cache_resource
def load_index():
    return faiss.read_index(INDEX_PATH)


@st.cache_resource
def load_metadata():
    return pd.read_parquet(METADATA_PATH)


@st.cache_resource
def load_bi_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_cross_encoder():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


@st.cache_resource
def load_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("❌ GROQ_API_KEY secret not set. Add it in Space Settings → Secrets.")
        st.stop()
    return Groq(api_key=api_key)


# ── Load everything ───────────────────────────
with st.spinner("⏳ Loading models (first run takes ~60s)..."):
    index        = load_index()
    metadata     = load_metadata()
    bi_encoder   = load_bi_encoder()
    cross_encoder = load_cross_encoder()
    groq_client  = load_groq_client()


# ── RAG Pipeline ──────────────────────────────
def retrieve(query: str, top_k: int = 50) -> pd.DataFrame:
    q_emb = bi_encoder.encode([query], normalize_embeddings=True).astype('float32')
    scores, indices = index.search(q_emb, top_k)
    results = metadata.iloc[indices[0]].copy()
    results['bi_encoder_score'] = scores[0]
    return results.reset_index(drop=True)


def rerank(query: str, candidates: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    pairs = [[query, row['abstract']] for _, row in candidates.iterrows()]
    scores = cross_encoder.predict(pairs)
    candidates = candidates.copy()
    candidates['rerank_score'] = scores
    return candidates.nlargest(top_k, 'rerank_score').reset_index(drop=True)


def generate_answer(query: str, top_chunks: pd.DataFrame) -> str:
    parts = []
    for i, (_, row) in enumerate(top_chunks.iterrows(), 1):
        parts.append(
            f"[{i}] Title: {row['title']}\n"
            f"    Abstract: {row['abstract']}"
        )
    context  = "\n\n".join(parts)
    user_msg = (
        f"Context Papers:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}\n\nAnswer (cite with [1],[2],...):"
    )
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/caduceus.png", width=60)
    st.title("Medical RAG System")
    st.markdown("---")

    st.markdown("### ⚙️ System Info")
    st.metric("📄 Papers Indexed", f"{len(metadata):,}")
    st.metric("🧠 Embedding Model", "MiniLM-L6-v2")
    st.metric("🤖 LLM", GROQ_MODEL.split("/")[-1])

    st.markdown("---")
    st.markdown("### 🎛️ Settings")
    top_k_retrieve = st.slider(
        "Retrieval candidates", 10, 100, 50, 10,
        help="Papers retrieved by bi-encoder before re-ranking"
    )
    top_k_final = st.slider(
        "Final sources shown", 1, 10, 5, 1,
        help="Papers shown after cross-encoder re-ranking"
    )

    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    examples = [
        "What deep learning methods detect lung cancer in CT scans?",
        "How effective are mRNA vaccines against COVID-19 variants?",
        "What biomarkers predict Alzheimer's disease progression?",
        "How does federated learning preserve patient privacy in medical AI?",
        "What neural networks are used for medical image segmentation?",
        "What are the latest treatments for Type 2 diabetes?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["query_input"] = ex

    st.markdown("---")
    st.caption("Built with FAISS · sentence-transformers · Groq · Streamlit")


# ── Main UI ───────────────────────────────────
st.title("🏥 Medical Research Q&A")
st.markdown(
    "Ask any medical or biomedical question. "
    "Answers are grounded in **100,000 arXiv medical papers** with cited sources."
)
st.markdown("---")

query = st.text_input(
    "🔍 Your Question",
    value=st.session_state.get("query_input", ""),
    placeholder="e.g. What deep learning methods are used for cancer detection?",
    key="query_input",
)

col1, col2 = st.columns([1, 6])
with col1:
    search_clicked = st.button("🚀 Search", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear"):
        st.session_state["query_input"] = ""
        st.rerun()

# ── Run pipeline ──────────────────────────────
if search_clicked and query.strip():

    with st.spinner("🔍 Retrieving relevant papers..."):
        candidates = retrieve(query, top_k=top_k_retrieve)

    with st.spinner("⚖️ Re-ranking with cross-encoder..."):
        top_chunks = rerank(query, candidates, top_k=top_k_final)

    with st.spinner("🤖 Generating answer..."):
        answer = generate_answer(query, top_chunks)

    # Answer
    st.markdown("## 📝 Answer")
    st.success(answer)

    # Source cards
    st.markdown("---")
    st.markdown(f"## 📚 Top {top_k_final} Sources")

    max_s = top_chunks["rerank_score"].max()
    min_s = top_chunks["rerank_score"].min()
    rng   = max_s - min_s if max_s != min_s else 1.0

    for i, (_, row) in enumerate(top_chunks.iterrows(), 1):
        norm = float((row["rerank_score"] - min_s) / rng)
        title_display = row['title'][:90] + ("..." if len(row['title']) > 90 else "")

        with st.expander(f"[{i}] {title_display}", expanded=(i == 1)):
            c1, c2, c3 = st.columns(3)
            c1.metric("Re-rank Score", f"{row['rerank_score']:.3f}")
            c2.metric("Year", str(row["year"]))
            c3.metric("Bi-encoder Score", f"{row['bi_encoder_score']:.3f}")

            st.progress(norm, text="Relevance")
            st.markdown("**Abstract:**")
            st.markdown(f"> {row['abstract']}")
            st.markdown(f"🔗 [View on arXiv]({row['arxiv_url']})")
            st.caption(f"Categories: {row['categories']}")

    # Pipeline details
    st.markdown("---")
    with st.expander("🔬 Pipeline Details"):
        st.markdown(f"""
| Step | Model | Result |
|---|---|---|
| **Bi-encoder retrieval** | `all-MiniLM-L6-v2` | {top_k_retrieve} candidates |
| **Cross-encoder re-rank** | `ms-marco-MiniLM-L-6-v2` | Top {top_k_final} selected |
| **Answer generation** | `{GROQ_MODEL}` | Grounded answer with citations |
        """)

    # History
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].insert(0, {
        "query":  query,
        "answer": answer[:200] + "...",
    })

elif search_clicked and not query.strip():
    st.warning("⚠️ Please enter a question before searching.")

# History panel
if st.session_state.get("history"):
    st.markdown("---")
    with st.expander(f"🕓 Query History ({len(st.session_state['history'])} queries)"):
        for item in st.session_state["history"]:
            st.markdown(f"**Q:** {item['query']}")
            st.markdown(f"**A:** {item['answer']}")
            st.markdown("---")

st.markdown("---")
st.caption(
    "⚠️ For research purposes only. "
    "Always consult a qualified medical professional for health decisions."
)
