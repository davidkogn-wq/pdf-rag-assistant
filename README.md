# pdf-rag-assistant
# 📚 Chat With Your PDF — Technical RAG Assistant

**[🚀 Click Here to Try the Live RAG App](https://pdf-rag-assistant-aoamjuc5f4d5dmvopaxecv.streamlit.app)**

A Retrieval-Augmented Generation (RAG) web application that allows engineers, legal teams, and business owners to upload dense PDF manuals, datasheets, or contracts and ask natural-language questions with **exact page citations** and verifiable source excerpts.

---

## 🎯 The Business Problem
Standard LLMs hallucinate technical specifications and cannot read private company documents out of the box. Meanwhile, engineers and support teams waste hours searching through 100-page PDF datasheets and SOPs using basic `Ctrl+F` keyword matching, which fails when terminology varies.

## 💡 How It Works Under the Hood
1. **PDF Parsing & Overlapping Chunking (`PyMuPDF`):** Extracts raw text page-by-page and splits documents into overlapping word windows so technical context is never cut in half across paragraph boundaries.
2. **Dense Vector Embeddings (`Sentence-Transformers`):** Encodes every chunk into a 384-dimensional vector using `all-MiniLM-L6-v2` and normalizes embeddings for exact cosine similarity search.
3. **In-Memory Vector Database (`FAISS`):** Stores document vectors in a fast `FAISS` (`IndexFlatIP`) index to retrieve the Top-K most semantically relevant paragraphs in milliseconds.
4. **Dual-Engine Grounded Answering:**
   * **Zero-Config Local Extractive QA (`DistilBERT-SQuAD2`):** Runs out-of-the-box on CPU to extract exact technical spans with confidence scores and page citations.
   * **Cloud LLM Synthesis (`Groq API`):** Supports optional one-click integration with Groq's high-speed LLM endpoint for conversational multi-chunk synthesis.

## 🛠️ Tech Stack
* **Vector Search & Embeddings:** `faiss-cpu`, `sentence-transformers` (`all-MiniLM-L6-v2`)
* **PDF Ingestion:** `PyMuPDF` (`fitz`)
* **QA & LLM Inference:** Hugging Face `transformers`, `Groq` API
* **Frontend UI:** Streamlit Chat Interface
