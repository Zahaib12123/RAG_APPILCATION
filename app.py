import os
import requests
import faiss
import gradio as gr
import numpy as np
import pandas as pd

from groq import Groq
from sentence_transformers import SentenceTransformer

# =========================
# API KEY (HUGGING FACE)
# =========================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY missing. Add it in Hugging Face → Settings → Secrets")

# =========================
# CONFIG
# =========================
LLM_MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4

CSV_URL = "https://drive.google.com/uc?id=1ncSsGZtsDwpn_xQAvNULioH4MEXC0Tj6"
CSV_PATH = "knowledge_base.csv"

# =========================
# INIT
# =========================
client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer(EMBED_MODEL)

documents = []
faiss_index = None

# =========================
# LOAD KNOWLEDGE BASE
# =========================
def load_csv_kb():
    global documents, faiss_index

    if not os.path.exists(CSV_PATH):
        r = requests.get(CSV_URL)
        with open(CSV_PATH, "wb") as f:
            f.write(r.content)

    df = pd.read_csv(CSV_PATH).fillna("")

    documents = [
        " | ".join([f"{col}: {row[col]}" for col in df.columns])
        for _, row in df.iterrows()
    ]

    embeddings = embedder.encode(
        documents,
        normalize_embeddings=True
    ).astype("float32")

    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)

    return f"✅ Knowledge base loaded ({len(documents)} rows)"

status = load_csv_kb()

# =========================
# RAG QA
# =========================
def answer_question(question: str):
    if not question.strip():
        return "⚠️ Please enter a valid question."

    query_emb = embedder.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = faiss_index.search(query_emb, TOP_K)
    retrieved_docs = [documents[i] for i in indices[0]]

    if not retrieved_docs:
        return "I don't know based on the provided data."

    context = "\n".join(retrieved_docs)

    prompt = f"""
You are a strict factual assistant.

Answer ONLY from the context.
If information is missing, say:
"I don't know based on the provided data."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return f"""
### ✅ Answer

{response.choices[0].message.content}
"""

# =========================
# UI (GEN-Z + HCI)
# =========================
css = """
body {
    background: linear-gradient(135deg,#020617,#1e1b4b);
    font-family: Inter, system-ui;
}
.gr-button {
    background: linear-gradient(90deg,#ec4899,#8b5cf6);
    color:white;
    font-weight:700;
    border-radius:14px;
}
textarea {
    background:#020617 !important;
    color:white !important;
    border-radius:14px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
        <h1 style="color:#ec4899">📊 CSV RAG Knowledge Assistant</h1>
        <p style="color:#c7d2fe">
        Ask questions strictly from the uploaded knowledge base
        </p>
        """
    )

    gr.Markdown(f"**Status:** {status}")

    question = gr.Textbox(
        label="Your Question",
        placeholder="Ask about sleep, stress, performance, trends...",
        lines=1
    )

    answer = gr.Markdown("🧠 Answer will appear here")

    gr.Button("Get Answer 🚀").click(
        answer_question,
        inputs=question,
        outputs=answer
    )

demo.launch()
