# AI Digital Clone – Haider Ali

**This is an AI clone, not the real person.**

## 1. Project Overview
This project is an AI-powered digital clone that answers questions about me using my personal knowledge base. It uses Hybrid Retrieval-Augmented Generation (RAG) to retrieve relevant information from my documents and generate answers in the first person.

## 2. Knowledge Base
The knowledge base contains six Markdown documents:

- `bio.md` – Personal biography and background.
- `cv.md` – Education, skills, and experience.
- `faq.md` – Frequently asked questions and answers.
- `opinions.md` – Personal opinions and preferences.
- `projects.md` – Academic and personal projects.
- `study_habits.md` – Study habits and learning methods.

The documents were divided into chunks of approximately 250 words with 50 words of overlap. Each chunk keeps its source information.

## 3. Retrieval Methods

| Method | Description |
|---|---|
| BM25 | Keyword-based retrieval using `rank_bm25`. |
| Dense Retrieval | Semantic retrieval using Sentence Transformers and cosine similarity. |
| Hybrid RAG | Combines BM25 and Dense rankings using Reciprocal Rank Fusion (RRF). |

The embedding model used is `sentence-transformers/all-MiniLM-L6-v2`. Embeddings are stored in `embeddings.npy` using NumPy.

## 4. Test Questions

The following questions are used to evaluate the retrieval methods:

| No. | Test Question | Expected Best Method |
|---|---|---|
| 1 | What is my educational background? | Hybrid |
| 2 | What are my main skills? | Hybrid |
| 3 | What is my approach to learning new concepts? | Dense |
| 4 | What is my final year project about?
