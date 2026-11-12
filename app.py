import os
import re
import glob
import html
import numpy as np
import streamlit as st

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


# =========================
# 1. SETTINGS
# =========================

DATA_DIR = "data"
EMBEDDING_FILE = "embeddings.npy"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
TOP_K = 5
FUSION_POOL = 10
RRF_K = 60

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "openai/gpt-oss-120b",
]

st.set_page_config(
    page_title="Haider Ali | AI Digital Clone",
    page_icon="🤖",
    layout="wide"
)


# =========================
# 2. DOCUMENTS AND CHUNKS
# =========================

def load_documents():
    documents = []
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.md")))

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            documents.append({
                "source": os.path.basename(file_path),
                "text": f.read()
            })

    return documents


def create_chunks(documents):
    chunks = []

    for doc in documents:
        words = doc["text"].split()
        start = 0
        chunk_id = 0

        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))

            chunks.append({
                "source": doc["source"],
                "chunk_id": chunk_id,
                "text": " ".join(words[start:end])
            })

            chunk_id += 1

            if end == len(words):
                break

            start = end - CHUNK_OVERLAP

    return chunks


# =========================
# 3. LOAD RAG SYSTEM
# =========================

@st.cache_resource
def load_rag_system():

    documents = load_documents()

    if not documents:
        raise RuntimeError(
            "No Markdown files found. Check that all six files are inside the data folder."
        )

    chunks = create_chunks(documents)

    embedding_model = SentenceTransformer(MODEL_NAME)

    if os.path.exists(EMBEDDING_FILE):
        embeddings = np.load(EMBEDDING_FILE)

        if embeddings.shape[0] != len(chunks):
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        embeddings = embedding_model.encode(
            [c["text"] for c in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        np.save(EMBEDDING_FILE, embeddings)

    bm25 = BM25Okapi([
        re.findall(r"\w+", c["text"].lower())
        for c in chunks
    ])

    chunk_index = {
        (c["source"], c["chunk_id"]): i
        for i, c in enumerate(chunks)
    }

    return documents, chunks, embedding_model, embeddings, bm25, chunk_index


# =========================
# 4. RETRIEVAL METHODS
# =========================

def tokenize(text):
    return re.findall(r"\w+", text.lower())


def build_results(indices, scores, chunks):
    results = []

    for rank, idx in enumerate(indices, start=1):
        results.append({
            "rank": rank,
            "source": chunks[idx]["source"],
            "chunk_id": chunks[idx]["chunk_id"],
            "text": chunks[idx]["text"],
            "score": float(scores[idx])
        })

    return results


def bm25_search(query, chunks, bm25, top_k=TOP_K):
    scores = bm25.get_scores(tokenize(query))
    indices = np.argsort(scores)[::-1][:top_k]

    return build_results(indices, scores, chunks)


def dense_search(query, chunks, embedding_model, embeddings, top_k=TOP_K):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    scores = np.dot(embeddings, query_embedding)
    indices = np.argsort(scores)[::-1][:top_k]

    return build_results(indices, scores, chunks)


def hybrid_search(
    query, chunks, bm25, embedding_model,
    embeddings, chunk_index, top_k=TOP_K
):
    bm25_results = bm25_search(
        query, chunks, bm25, FUSION_POOL
    )

    dense_results = dense_search(
        query, chunks, embedding_model, embeddings, FUSION_POOL
    )

    rrf_scores = {}

    for results in (bm25_results, dense_results):
        for result in results:
            key = (result["source"], result["chunk_id"])

            rrf_scores[key] = (
                rrf_scores.get(key, 0)
                + 1 / (RRF_K + result["rank"])
            )

    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    final_results = []

    for rank, (key, score) in enumerate(
        sorted_results[:top_k], start=1
    ):
        chunk = chunks[chunk_index[key]]

        final_results.append({
            "rank": rank,
            "source": key[0],
            "chunk_id": key[1],
            "text": chunk["text"],
            "score": float(score)
        })

    return final_results


# =========================
# 5. LLM ANSWER GENERATION
# =========================

def generate_answer(question, results):

    context = "\n\n".join(
        f"Source: {r['source']} (chunk {r['chunk_id']})\n{r['text']}"
        for r in results
    )

    system_prompt = """
You are the AI clone of Haider Ali.

Speak in first person, using a friendly and simple tone.

RULES:
1. Answer ONLY using the CONTEXT provided.
2. If the context does not clearly contain the answer, reply exactly:
   I don't have that in my notes.
3. Never guess or invent facts about Haider.
4. Never use general knowledge to answer questions about Haider.
5. Never share phone numbers, addresses, ID numbers or private details.
6. If asked whether you are the real person, say you are an AI clone of Haider Ali.
7. Keep answers short and natural.
"""

    user_prompt = f"""
CONTEXT:
{context}

QUESTION:
{question}
"""

    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        return (
            "LLM API key is not configured. "
            "An HF_TOKEN secret is needed for generated answers."
        )

    client = InferenceClient(token=hf_token)
    last_error = None

    for model_name in LLM_MODELS:
        try:
            response = client.chat_completion(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.2
            )

            return response.choices[0].message.content

        except Exception as e:
            last_error = e
            print(f"{model_name} failed: {e}")

    return f"LLM error: {str(last_error)}"


# =========================
# 6. STREAMLIT INTERFACE
# =========================

st.markdown(
    """
    <div style="
        background-color:#0f172a;
        padding:28px;
        border-radius:12px;
        color:white;
    ">
        <h1 style="color:white;">HA | Haider Ali</h1>
        <p>AI Digital Clone · Hybrid RAG (BM25 + Dense + RRF)</p>
        <div style="
            background:#78350f;
            padding:10px;
            border-radius:6px;
            color:#fde68a;
            font-weight:bold;
        ">
            This is an AI clone, not the real person.
        </div>
        <p>BSSE · UMT · Sialkot, Pakistan</p>
        <p>Ask a question about me. Answers are based on my own notes.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

try:
    with st.spinner("Loading documents and retrieval models..."):
        (
            documents,
            chunks,
            embedding_model,
            embeddings,
            bm25,
            chunk_index
        ) = load_rag_system()

except Exception as e:
    st.error(f"Could not load the knowledge base: {e}")
    st.stop()


st.success(
    f"Knowledge base loaded: {len(documents)} documents "
    f"and {len(chunks)} chunks."
)

st.subheader("Ask me a question")

retrieval_mode = st.radio(
    "Choose retrieval method",
    ["BM25 only", "Dense only", "Hybrid"],
    index=2,
    horizontal=True
)

question = st.text_input(
    "Your question",
    placeholder="For example: What is my final year project?"
)

with st.expander("Example questions"):
    st.write("""
    - What is ECHO TB?
    - Tell me about your PPE detection project.
    - Do you have a job or business?
    - How do you study new technologies?
    - What do you think about AI?
    - Do you drink coffee?
    """)

ask_button = st.button("Ask", type="primary")

if ask_button:

    if not question.strip():
        st.warning("Please type a question first.")

    else:
        with st.spinner("Searching your notes and preparing an answer..."):

            if retrieval_mode == "BM25 only":
                results = bm25_search(
                    question, chunks, bm25
                )

            elif retrieval_mode == "Dense only":
                results = dense_search(
                    question, chunks, embedding_model, embeddings
                )

            else:
                results = hybrid_search(
                    question,
                    chunks,
                    bm25,
                    embedding_model,
                    embeddings,
                    chunk_index
                )

            answer = generate_answer(question, results)

        st.subheader("Answer")

        if answer.startswith(("LLM error", "LLM API key")):
            st.warning(answer)
        else:
            st.markdown(answer)

        st.subheader("Retrieved passages, sources and scores")

        for result in results:
            with st.expander(
                f"#{result['rank']} | {result['source']} | "
                f"Chunk {result['chunk_id']} | Score: {result['score']:.4f}",
                expanded=(result["rank"] == 1)
            ):
                st.write(result["text"])


st.divider()

st.caption(
    "Built with Python, Streamlit, BM25, Sentence Transformers, "
    "NumPy and Reciprocal Rank Fusion. "
    "This is an AI clone, not the real person."
)