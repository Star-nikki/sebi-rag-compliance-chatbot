# SEBI-RAG: AI Compliance Assistant for SEBI LODR 2026

A multi-model Generative AI framework that answers questions about the **SEBI Listing Obligations and Disclosure Requirements (LODR) Master Circular 2026** — a 291-page Indian regulatory document — using RAG, fine-tuning, and hybrid retrieval.

---

## Three Chatbot Systems

| System | Model | Runs On | Fine-tuned |
|--------|-------|---------|------------|
| **SEBI Sovereign** | Llama-3.2-3B (LoRA fine-tuned) | GPU | Yes |
| **Qwen RAG Chatbot** | Qwen2.5-1.5B-Instruct | CPU only | No |
| **DPDP Chatbot** | GPT-2 | CPU only | No |

---

## Project Structure

```
sebi_sovereign/        # Main app: Llama fine-tuned + hybrid BM25+FAISS retrieval
sebi_streamlit_app/    # Qwen2.5-1.5B chatbot: 100% local, no API key needed
sebi_chatbot/          # FastAPI backend + Streamlit frontend (Llama)
sebi_rag_pure/         # Pure RAG baseline (no fine-tuning)
dpdp_chatbot/          # DPDP Act 2023 + RBI banking regulations chatbot
Step 1.ipynb           # Data ingestion, chunking, EDA, vector indexing
Step 2.ipynb           # RAFT dataset generation using LLaMA-3.1-8B via Groq
SEBI_Finetune_Colab.ipynb  # LoRA fine-tuning on Google Colab T4 GPU
sebi_local_train.jsonl # 1,125 RAFT-generated Q&A pairs
```

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/sebi-rag-compliance-chatbot.git
cd sebi-rag-compliance-chatbot
```

### 2. Set up environment variables
```bash
cp .env.example .env
# Edit .env and fill in your HF_TOKEN and GROQ_API_KEY
```

### 3. Run the Qwen chatbot (no GPU needed)
```bash
cd sebi_streamlit_app
pip install -r requirements.txt
streamlit run app.py
# First run downloads Qwen2.5-1.5B (~3GB) and builds FAISS index (~2 min)
# After that, instant startup
```

### 4. Run SEBI Sovereign (requires GPU)
```bash
cd sebi_sovereign
pip install -r requirements.txt
streamlit run app.py
```

### 5. Run the DPDP chatbot
```bash
cd dpdp_chatbot
pip install -r requirements.txt  # see requirements.md
python create_db.py               # builds the FAISS index from PDFs
streamlit run app.py
```

---

## Fine-Tuned Model

The Llama-3.2-3B model fine-tuned on the SEBI RAFT dataset is available on Hugging Face:

**[NikkiRani/sebi-llama-3b](https://huggingface.co/NikkiRani/sebi-llama-3b)**

Fine-tuning was done using LoRA (rank 16) + 4-bit quantization via Unsloth on a free Google Colab T4 GPU. See `SEBI_Finetune_Colab.ipynb` to reproduce.

---

## Dataset

`sebi_local_train.jsonl` contains 1,125 question-answer pairs synthesized from the SEBI LODR 2026 using LLaMA-3.1-8B-Instant via the Groq API. It covers 88.7% of the circular's 291 pages. See `Step 2.ipynb` for the full generation pipeline.

---

## Tech Stack

- **Models:** Llama-3.2-3B-Instruct, Qwen2.5-1.5B-Instruct, LLaMA-3.1-8B (synthesis)
- **Fine-tuning:** Unsloth + LoRA + 4-bit NF4 quantization + TRL SFTTrainer
- **Retrieval:** FAISS + BM25 Okapi hybrid (60/40 weighted fusion)
- **Embeddings:** all-MiniLM-L6-v2 (sentence-transformers)
- **Framework:** LangChain, Streamlit, FastAPI
- **Vector stores:** ChromaDB (dev), FAISS (production)

---

## Important Notes

- The SEBI LODR 2026 PDF is **not included** in this repository due to copyright. Download it directly from [sebi.gov.in](https://www.sebi.gov.in).
- Never commit your `.env` file. Use `.env.example` as a template.
- The FAISS index files (`.faiss`, `.pkl`) are auto-generated on first run and are excluded from version control.
- Outputs of this system are for informational purposes only and do not constitute legal advice.
