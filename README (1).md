# DocMind — a Retrieval-Augmented Generation (RAG) assistant

Upload a document, ask it questions in plain English, and get answers
grounded in that document's actual content — with the source passages
shown alongside the answer, not hidden.

This was built specifically to demonstrate hands-on RAG experience for
job applications that ask for it.

## How it works (the actual RAG pipeline)

```
 Upload                         Ask
   │                              │
   ▼                              ▼
Extract text              Embed the question
(pypdf for PDFs)          (same model as docs)
   │                              │
   ▼                              │
Chunk into ~800-char              │
overlapping pieces                │
   │                              │
   ▼                              │
Embed each chunk                  │
(sentence-transformers,           │
 local, free, no API)             │
   │                              │
   ▼                              ▼
          Vector store (numpy cosine similarity)
                       │
                       ▼
          Top-4 most relevant chunks
                       │
                       ▼
     LLM generates an answer using ONLY those
     chunks as context (Anthropic / OpenAI / or
     "retrieval-only" if no API key is set)
                       │
                       ▼
          Answer + cited source passages
```

**Retrieval** = finding the right passages. **Generation** = writing a
fluent answer from them. RAG is the combination, and keeping these two
steps separate (as this code does — `rag/store.py` vs `rag/generator.py`)
is the single most important design idea to be able to explain in an
interview.

## Project structure
```
docmind-rag/
├── app.py                 # Flask routes: /, /upload, /ask, /reset
├── requirements.txt
├── .env.example            # copy to .env and add an API key (optional)
├── rag/
│   ├── chunking.py         # PDF/text extraction + overlapping chunking
│   ├── store.py            # embeddings + vector store + cosine search
│   └── generator.py        # LLM call, with graceful no-key fallback
├── templates/index.html
└── static/{app.js, style.css}
```

## Run it locally

```bash
cd docmind-rag
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000

**First run note:** the first time you upload a document, it downloads
the embedding model (~80MB, one-time, needs internet). After that it's
fully local and free.

**No API key needed to try it.** Without an `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY` in a `.env` file, DocMind runs in "retrieval-only"
mode — it shows you the most relevant passages directly instead of a
generated answer. Add either key (copy `.env.example` to `.env`) to get
full generated answers.

## Try it
1. Upload a `.txt` or `.pdf` (a resume, a course PDF, a README — anything)
2. Ask a question only that document could answer
3. Notice the answer cites which passage it came from, and that asking
   something the document doesn't cover gets an honest "not found"
   rather than a hallucinated answer

## What to say about this in an interview

- **"Why chunking with overlap?"** — if a sentence that answers a
  question gets split across two chunks, overlap increases the odds at
  least one chunk has the full idea intact.
- **"Why cosine similarity on normalized vectors?"** — it reduces to a
  simple dot product, and normalizing means document length doesn't
  bias the score.
- **"Why not just use FAISS/Pinecone from the start?"** — building
  retrieval with plain numpy first forces you to actually understand
  what a vector database optimizes (approximate nearest-neighbor search
  at scale) before reaching for the tool that does it for you.
- **"What's the weakest part of this system?"** — fixed-size chunking
  is naive; it can split a table or a list mid-thought. A stronger
  version would chunk by semantic boundaries (paragraphs, headers) and
  re-rank retrieved chunks with a cross-encoder before generation.

## Ideas to extend it (good for a "what would you add" question)
- Swap in FAISS or pgvector for the vector store at scale
- Add a re-ranking step (cross-encoder) after initial retrieval
- Support multi-document source filtering ("only search doc X")
- Turn this into an **agentic** workflow: let the model decide whether
  it needs to search again, ask a clarifying question, or call another
  tool before answering — this is the natural next step from RAG to
  "agentic AI workflows."
