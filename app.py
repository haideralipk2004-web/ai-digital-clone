import os
import re
import html
import glob
import numpy as np
import gradio as gr

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


# =========================================================
# 1. SETTINGS
# =========================================================

DATA_DIR = "data"
EMBEDDING_FILE = "embeddings.npy"

CHUNK_SIZE = 250       # words per chunk
CHUNK_OVERLAP = 50     # words shared between neighbouring chunks
TOP_K = 5              # chunks shown and sent to the LLM
FUSION_POOL = 10       # results each retriever gives to the fusion step
RRF_K = 60             # RRF constant: 1 / (60 + rank)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "openai/gpt-oss-120b",
]


# =========================================================
# 2. LOAD DOCUMENTS
# =========================================================

def load_documents():
    documents = []
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.md")))

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        documents.append({
            "source": os.path.basename(file_path),
            "text": text
        })

    return documents


# =========================================================
# 3. CHUNKING
# =========================================================

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


documents = load_documents()
chunks = create_chunks(documents)

chunk_index = {
    (c["source"], c["chunk_id"]): i for i, c in enumerate(chunks)
}

print(f"Loaded {len(documents)} documents")
print(f"Created {len(chunks)} chunks")


# =========================================================
# 4. BM25 INDEX
# =========================================================

def tokenize(text):
    return re.findall(r"\w+", text.lower())


bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])


# =========================================================
# 5. DENSE MODEL + EMBEDDINGS (precomputed, saved as .npy)
# =========================================================

embedding_model = SentenceTransformer(MODEL_NAME)

embeddings = None

if os.path.exists(EMBEDDING_FILE):
    print("Loading saved embeddings...")
    embeddings = np.load(EMBEDDING_FILE)

    if embeddings.shape[0] != len(chunks):
        print("Saved embeddings do not match the chunks. Recreating...")
        embeddings = None

if embeddings is None:
    print("Creating embeddings...")
    embeddings = embedding_model.encode(
        [c["text"] for c in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    np.save(EMBEDDING_FILE, embeddings)
    print("Embeddings saved.")


# =========================================================
# 6. RETRIEVERS
# =========================================================

def build_results(indices, scores):
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


def bm25_search(query, top_k=TOP_K):
    scores = bm25.get_scores(tokenize(query))
    top_indices = np.argsort(scores)[::-1][:top_k]
    return build_results(top_indices, scores)


def dense_search(query, top_k=TOP_K):
    query_embedding = embedding_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]

    # Embeddings are normalized, so dot product = cosine similarity
    scores = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return build_results(top_indices, scores)


# Hybrid: Reciprocal Rank Fusion, score = sum of 1 / (60 + rank)
def hybrid_search(query, top_k=TOP_K):
    bm25_results = bm25_search(query, top_k=FUSION_POOL)
    dense_results = dense_search(query, top_k=FUSION_POOL)

    rrf_scores = {}

    for results in (bm25_results, dense_results):
        for r in results:
            key = (r["source"], r["chunk_id"])
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + r["rank"])

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    final_results = []
    for rank, (key, score) in enumerate(sorted_results[:top_k], start=1):
        chunk = chunks[chunk_index[key]]
        final_results.append({
            "rank": rank,
            "source": key[0],
            "chunk_id": key[1],
            "text": chunk["text"],
            "score": float(score)
        })

    return final_results


# =========================================================
# 7. LLM
# =========================================================

NO_TOKEN_MESSAGE = (
    "LLM API key is not configured. Set the HF_TOKEN environment variable "
    "(locally) or add HF_TOKEN as a secret in the Space settings."
)


def generate_answer(question, retrieved_results):

    context = "\n\n".join(
        f"Source: {r['source']} (chunk {r['chunk_id']})\n{r['text']}"
        for r in retrieved_results
    )

    system_prompt = (
        "You are the AI clone of Haider Ali, a 21-year-old BSSE student from Sialkot. "
        "Speak in the first person, in a friendly and simple tone, using short sentences.\n\n"
        "RULES:\n"
        "1. Answer ONLY using the CONTEXT provided by the user message. It comes from my own documents.\n"
        "2. If the context does not clearly contain the answer, reply exactly: "
        "\"I don't have that in my notes.\" Do not guess, do not use general knowledge, "
        "and do not invent facts about me. This applies to any question that is not about me.\n"
        "3. Never share phone numbers, addresses, ID numbers or other private details.\n"
        "4. If asked whether you are the real person, say you are an AI clone of Haider Ali.\n"
        "5. Keep the answer short and natural."
    )

    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        return NO_TOKEN_MESSAGE

    client = InferenceClient(token=hf_token)
    last_error = None

    for model_name in LLM_MODELS:
        try:
            response = client.chat_completion(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=300,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            print(f"Model {model_name} failed: {e}")

    return f"LLM error: {str(last_error)}"


# =========================================================
# 8. UI HELPERS
# =========================================================

ACCENT = "#2563eb"

MODE_COLORS = {
    "BM25 only": "#0f766e",
    "Dense only": "#b45309",
    "Hybrid": "#2563eb",
}

GLOBAL_CSS = """
<style>
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
footer { visibility: hidden; }
.card-title { font-size: 13px; letter-spacing: .06em; text-transform: uppercase;
              color: var(--body-text-color-subdued); font-weight: 600; }
</style>
"""

CARD_STYLE = (
    "background:var(--background-fill-secondary);"
    "border:1px solid var(--border-color-primary);"
    "border-radius:10px;padding:18px 20px;"
    "color:var(--body-text-color);"
)

HERO_HTML = """
<div style="background:#0f172a;border-radius:12px;padding:28px 32px;color:#f8fafc;">
  <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">
    <div style="width:60px;height:60px;border-radius:12px;background:#2563eb;display:flex;
                align-items:center;justify-content:center;font-size:22px;font-weight:700;
                color:#ffffff;letter-spacing:1px;">HA</div>
    <div>
      <div style="font-size:28px;font-weight:700;line-height:1.2;color:#f8fafc;">Haider Ali</div>
      <div style="font-size:14px;color:#94a3b8;margin-top:2px;">
        Digital Clone &nbsp;|&nbsp; Hybrid RAG: BM25, dense embeddings and Reciprocal Rank Fusion
      </div>
    </div>
  </div>
  <div style="margin-top:18px;padding:10px 14px;border-left:4px solid #f59e0b;
              background:rgba(245,158,11,0.12);color:#fde68a;font-weight:600;font-size:14px;
              border-radius:0 6px 6px 0;">
    This is an AI clone, not the real person.
  </div>
  <div style="margin-top:16px;display:flex;gap:8px;flex-wrap:wrap;">
    <span style="border:1px solid #334155;padding:4px 12px;border-radius:6px;font-size:13px;color:#cbd5e1;">BSSE, UMT</span>
    <span style="border:1px solid #334155;padding:4px 12px;border-radius:6px;font-size:13px;color:#cbd5e1;">Sialkot, Pakistan</span>
    <span style="border:1px solid #334155;padding:4px 12px;border-radius:6px;font-size:13px;color:#cbd5e1;">AI and Machine Learning</span>
  </div>
  <div style="margin-top:14px;font-size:14px;color:#94a3b8;">
    Ask a question about me. Answers are generated only from my own notes, and the passages used are shown below.
  </div>
</div>
"""

EMPTY_ANSWER_HTML = f"""
<div style="{CARD_STYLE}min-height:190px;display:flex;flex-direction:column;justify-content:center;">
  <div class="card-title">Answer</div>
  <div style="font-size:16px;margin-top:8px;color:var(--body-text-color-subdued);">
    The answer will appear here. Type a question, choose a retrieval mode and press Ask.
  </div>
</div>
"""

EMPTY_CHUNKS_HTML = f"""
<div style="{CARD_STYLE}color:var(--body-text-color-subdued);font-size:14px;">
  The passages used to answer, with source file, chunk ID and score, will be listed here.
</div>
"""

LOADING_HTML = f"""
<div style="{CARD_STYLE}min-height:190px;display:flex;flex-direction:column;justify-content:center;">
  <div class="card-title">Answer</div>
  <div style="font-size:16px;margin-top:8px;color:{ACCENT};font-weight:600;">
    Searching my notes and preparing the answer...
  </div>
</div>
"""

FOOTER_HTML = """
<div style="text-align:center;font-size:13px;color:var(--body-text-color-subdued);padding:12px 0 4px 0;line-height:1.6;">
  Built with BM25, Sentence-Transformers (all-MiniLM-L6-v2), Reciprocal Rank Fusion and Gradio.<br>
  Answers come only from Haider Ali's notes. If something is not in the notes, the clone says so.
</div>
"""


def section_title(text):
    return (
        "<div style=\"font-size:18px;font-weight:700;margin:10px 0 2px 2px;"
        f"color:var(--body-text-color);\">{html.escape(text)}</div>"
    )


def render_message(text, color="#f59e0b"):
    return (
        f"<div style=\"{CARD_STYLE}border-left:4px solid {color};\">"
        f"{html.escape(text)}</div>"
    )


def render_answer(answer, mode, results):

    safe_answer = html.escape(answer).replace("\n", "<br>")

    is_problem = answer.startswith("LLM error") or answer.startswith("LLM API key")
    border = "#d97706" if is_problem else ACCENT
    heading = "Notice" if is_problem else "Answer from Haider's AI clone"

    sources = []
    for r in results:
        label = f"{r['source']} #{r['chunk_id']}"
        if label not in sources:
            sources.append(label)

    chips = "".join(
        "<span style=\"background:var(--background-fill-primary);"
        "border:1px solid var(--border-color-primary);padding:3px 10px;border-radius:6px;"
        f"font-size:12px;margin-right:6px;display:inline-block;margin-bottom:6px;\">{html.escape(s)}</span>"
        for s in sources
    )

    color = MODE_COLORS.get(mode, ACCENT)

    return f"""
<div style="{CARD_STYLE}border-left:4px solid {border};">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
    <div class="card-title">{heading}</div>
    <span style="background:{color};color:#ffffff;padding:3px 12px;border-radius:6px;
                 font-size:12px;font-weight:600;">{html.escape(mode)}</span>
  </div>
  <div style="font-size:17px;line-height:1.65;margin-top:12px;">{safe_answer}</div>
  <div class="card-title" style="margin-top:18px;">Sources used</div>
  <div style="margin-top:8px;">{chips}</div>
</div>
"""


def render_chunks(results, mode):

    if not results:
        return EMPTY_CHUNKS_HTML

    max_score = max(r["score"] for r in results)
    color = MODE_COLORS.get(mode, ACCENT)

    cards = []

    for i, r in enumerate(results):

        if max_score > 0 and r["score"] > 0:
            width = max(4, int(100 * r["score"] / max_score))
        else:
            width = 4

        open_attr = "open" if i == 0 else ""

        cards.append(f"""
<details {open_attr} style="{CARD_STYLE}margin-bottom:10px;">
  <summary style="cursor:pointer;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
    <span style="background:{color};color:#ffffff;border-radius:6px;padding:2px 10px;font-weight:700;">#{r['rank']}</span>
    <span style="font-weight:600;">{html.escape(r['source'])}</span>
    <span style="color:var(--body-text-color-subdued);font-size:13px;">chunk {r['chunk_id']}</span>
    <span style="margin-left:auto;font-weight:600;font-size:14px;">score {r['score']:.4f}</span>
  </summary>
  <div style="height:5px;border-radius:999px;background:rgba(100,116,139,0.25);margin:12px 0;">
    <div style="width:{width}%;height:100%;border-radius:999px;background:{color};"></div>
  </div>
  <div style="font-size:14px;line-height:1.6;color:var(--body-text-color);">{html.escape(r['text'])}</div>
</details>
""")

    return "".join(cards)


# =========================================================
# 9. MAIN FUNCTION CALLED BY THE UI
# =========================================================

def answer_question(question, retrieval_mode):

    if not question or not question.strip():
        yield render_message("Please type a question first."), EMPTY_CHUNKS_HTML
        return

    yield LOADING_HTML, EMPTY_CHUNKS_HTML

    if retrieval_mode == "BM25 only":
        results = bm25_search(question)
    elif retrieval_mode == "Dense only":
        results = dense_search(question)
    else:
        results = hybrid_search(question)

    answer = generate_answer(question, results)

    yield (
        render_answer(answer, retrieval_mode, results),
        render_chunks(results, retrieval_mode)
    )


def clear_all():
    return "", EMPTY_ANSWER_HTML, EMPTY_CHUNKS_HTML


# =========================================================
# 10. GRADIO UI
# =========================================================

EXAMPLE_QUESTIONS = [
    "What is ECHO TB?",
    "Tell me about your PPE detection project.",
    "Do you have a job or business?",
    "How do you study new technologies?",
    "What do you think about AI?",
    "Do you drink coffee?",
    "Who won the 2022 football world cup?",
]

with gr.Blocks(title="Haider Ali | AI Digital Clone") as demo:

    gr.HTML(GLOBAL_CSS)
    gr.HTML(HERO_HTML)

    with gr.Row(equal_height=False):

        with gr.Column(scale=2):

            question = gr.Textbox(
                label="Your question",
                placeholder="For example: What is ECHO TB?",
                lines=3
            )

            retrieval_mode = gr.Radio(
                choices=["BM25 only", "Dense only", "Hybrid"],
                value="Hybrid",
                label="Retrieval mode",
                info="BM25 = keyword match | Dense = meaning match | Hybrid = both combined with RRF"
            )

            with gr.Row():
                submit_button = gr.Button("Ask", variant="primary", scale=3)
                clear_button = gr.Button("Clear", scale=1)

            gr.Examples(
                examples=[[q] for q in EXAMPLE_QUESTIONS],
                inputs=[question],
                label="Example questions"
            )

        with gr.Column(scale=3):
            answer_box = gr.HTML(EMPTY_ANSWER_HTML)

    gr.HTML(section_title("Retrieved passages, sources and scores"))
    chunks_box = gr.HTML(EMPTY_CHUNKS_HTML)
    gr.HTML(FOOTER_HTML)

    submit_button.click(
        fn=answer_question,
        inputs=[question, retrieval_mode],
        outputs=[answer_box, chunks_box]
    )

    question.submit(
        fn=answer_question,
        inputs=[question, retrieval_mode],
        outputs=[answer_box, chunks_box]
    )

    clear_button.click(
        fn=clear_all,
        inputs=None,
        outputs=[question, answer_box, chunks_box]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )