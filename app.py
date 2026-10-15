import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from groq import Groq

# 1. Page Configuration
st.set_page_config(
    page_title="Chat With Your PDF (RAG Assistant)",
    page_icon="📚",
    layout="wide"
)

# 2. Load Embedding & Local QA Models (Cached in memory)
@st.cache_resource
def load_rag_models():
    # Compact ~80MB embedding model for semantic vector search
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    # Built-in Extractive QA model so the app works immediately without requiring an API key
    qa_pipe = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )
    return embedder, qa_pipe

embedder, qa_pipe = load_rag_models()

# 3. Built-in Sample Technical Datasheet (For 1-Click Recruiter Testing)
def get_sample_datasheet_pages():
    return [
        {
            "page": 1,
            "text": (
                "ESP32-S3-IND Industrial Motor & Telemetry Controller Datasheet (Rev 4.2). "
                "1. General Overview: The ESP32-S3-IND is a ruggedized dual-core microcontroller module "
                "designed for industrial automation, high-frequency PWM motor control, and real-time sensor telemetry. "
                "The processor operates at an adjustable clock frequency from 80 MHz up to 240 MHz and includes "
                "512 KB of internal SRAM and 8 MB of Octal SPI Flash."
            )
        },
        {
            "page": 4,
            "text": (
                "Section 4: Pinout & Electrical Specifications. "
                "Supply Rail (VDD_3V3): Nominal operating voltage is 3.3V DC (recommended range 3.0V to 3.6V). "
                "Pin 4 (GPIO4 / ADC1_CH3 - Analog Sensor Input): The maximum operating voltage on Pin 4 is 3.6V DC. "
                "Applying more than 3.6V to Pin 4 will permanently damage the internal CMOS protection diodes. "
                "Maximum continuous sink/source current per GPIO pin is 40 mA."
            )
        },
        {
            "page": 7,
            "text": (
                "Section 7: Thermal Management & Overheating Protection. "
                "The industrial operating temperature range is -40°C to +85°C ambient. "
                "The internal silicon junction includes an automated hardware thermal shutdown circuit. "
                "Thermal shutdown triggers immediately when the core temperature exceeds 115°C, disabling all PWM "
                "motor driver outputs until the die cools below 95°C."
            )
        },
        {
            "page": 11,
            "text": (
                "Section 11: I2C & SPI Peripheral Interfaces. "
                "The hardware I2C controller supports Standard Mode (100 kHz) and Fast Mode (400 kHz) clock frequencies. "
                "External 4.7 kOhm pull-up resistors are required on both the SDA (Pin 8) and SCL (Pin 9) lines "
                "for reliable high-speed communication with external accelerometers and thermistors."
            )
        },
        {
            "page": 15,
            "text": (
                "Section 15: Warranty, Compliance & Return Policy. "
                "All ESP32-S3-IND industrial modules carry a 24-month limited hardware warranty against manufacturing defects. "
                "Damage caused by reverse-polarity power connections, electrostatic discharge (ESD) above 2 kV, or exceeding "
                "the 3.6V absolute maximum pin rating voids the standard warranty."
            )
        }
    ]

# 4. Document Chunking & FAISS Indexing Functions
def chunk_pages(pages_data, chunk_size=180, overlap=40):
    chunks = []
    for item in pages_data:
        page_num = item["page"]
        words = item["text"].split()
        if not words:
            continue
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            if len(chunk_text.strip()) > 20:
                chunks.append({
                    "page": page_num,
                    "text": chunk_text
                })
    return chunks

def build_faiss_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # Normalize vectors so Inner Product equals Cosine Similarity
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def parse_uploaded_pdf(uploaded_file):
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_data = []
    for page_idx in range(len(doc)):
        text = doc[page_idx].get_text("text").strip()
        if text:
            pages_data.append({"page": page_idx + 1, "text": text})
    return pages_data

# 5. Sidebar: PDF Upload, Sample Loader & Optional LLM API Key
st.sidebar.header("📂 1. Document Source")
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF Manual / Contract", type=["pdf"])

if st.sidebar.button("📄 Load Sample ESP32 Datasheet", use_container_width=True):
    pages = get_sample_datasheet_pages()
    chunks = chunk_pages(pages)
    st.session_state["chunks"] = chunks
    st.session_state["faiss_index"] = build_faiss_index(chunks)
    st.session_state["doc_name"] = "ESP32-S3-IND_Hardware_Datasheet.pdf (Sample)"
    st.session_state["messages"] = []

if uploaded_pdf is not None:
    if st.session_state.get("doc_name") != uploaded_pdf.name:
        with st.spinner("Extracting text, chunking, and building FAISS vector index..."):
            pages = parse_uploaded_pdf(uploaded_pdf)
            if not pages:
                st.sidebar.error("Could not extract text from this PDF (it may be a scanned image).")
            else:
                chunks = chunk_pages(pages)
                st.session_state["chunks"] = chunks
                st.session_state["faiss_index"] = build_faiss_index(chunks)
                st.session_state["doc_name"] = uploaded_pdf.name
                st.session_state["messages"] = []

st.sidebar.divider()
st.sidebar.header("🤖 2. Answer Engine")
groq_api_key = st.sidebar.text_input(
    "Optional Groq API Key (Llama / GPT-OSS):",
    type="password",
    help="Leave blank to use the built-in local Hugging Face Extractive QA model, or paste a free key from console.groq.com for full generative answers."
)

top_k = st.sidebar.slider("Retrieved Context Chunks (Top-K)", min_value=1, max_value=5, value=3)

# Auto-load sample datasheet on very first visit so the app is immediately testable
if "chunks" not in st.session_state:
    pages = get_sample_datasheet_pages()
    chunks = chunk_pages(pages)
    st.session_state["chunks"] = chunks
    st.session_state["faiss_index"] = build_faiss_index(chunks)
    st.session_state["doc_name"] = "ESP32-S3-IND_Hardware_Datasheet.pdf (Sample)"
    st.session_state["messages"] = []

# 6. Main UI Header & Document Status Bar
st.title("📚 Chat With Your PDF — Technical RAG Assistant")
st.markdown("Ask questions about dense datasheets, contracts, or textbooks. Every answer is grounded strictly in the uploaded document with **exact page citations**.")

m1, m2, m3 = st.columns(3)
m1.metric("Active Document", st.session_state["doc_name"][:28])
m2.metric("Indexed Vector Chunks", len(st.session_state["chunks"]))
m3.metric("Active Engine", "Groq Cloud LLM" if groq_api_key else "Local Extractive QA (Zero-Key)")

st.divider()

# 7. Quick-Test Question Buttons
st.caption("💡 **One-Click Test Questions (for the Sample ESP32 Datasheet):**")
q_col1, q_col2, q_col3 = st.columns(3)
preset_query = None

if q_col1.button("⚡ Max voltage on Pin 4?", use_container_width=True):
    preset_query = "What is the maximum operating voltage on Pin 4?"
if q_col2.button("🌡️ When does thermal shutdown trigger?", use_container_width=True):
    preset_query = "At what temperature does thermal shutdown trigger?"
if q_col3.button("🔌 What pull-up resistors are needed for I2C?", use_container_width=True):
    preset_query = "What pull-up resistors and clock frequencies are supported for I2C?"

# 8. RAG Retrieval + Answer Generation Function
def answer_question(query):
    chunks = st.session_state["chunks"]
    index = st.session_state["faiss_index"]

    # Embed query & search FAISS
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    k = min(top_k, len(chunks))
    scores, indices = index.search(q_emb, k)

    retrieved = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            retrieved.append({
                "page": chunks[idx]["page"],
                "text": chunks[idx]["text"],
                "score": float(score)
            })

    # Path A: If user supplied a Groq API Key, use Generative LLM
    if groq_api_key.strip():
        try:
            client = Groq(api_key=groq_api_key.strip())
            context_str = "\n\n".join([f"[Page {r['page']}]: {r['text']}" for r in retrieved])
            system_prompt = (
                "You are a precise technical assistant. Answer the user's question using ONLY the provided context. "
                "Always cite the exact page number in brackets like [Page 4]. If the answer is not in the context, state that clearly."
            )
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
                ],
                temperature=0.1
            )
            answer_text = completion.choices[0].message.content
            return answer_text, retrieved
        except Exception as e:
            st.warning(f"Groq API error ({e}). Falling back to built-in Local QA model.")

    # Path B: Built-in Zero-Key Extractive QA Model
    best_ans = None
    best_conf = -1.0
    best_chunk = retrieved[0]

    for r in retrieved:
        res = qa_pipe(question=query, context=r["text"])
        if res["score"] > best_conf:
            best_conf = res["score"]
            best_ans = res["answer"]
            best_chunk = r

    answer_text = (
        f"**Direct Answer:** {best_ans} *( cited from **[Page {best_chunk['page']}]** | "
        f"Confidence: `{best_conf:.2f}` )*\n\n"
        f"> **Supporting Excerpt [Page {best_chunk['page']}]:** \"{best_chunk['text']}\""
    )
    return answer_text, retrieved

# 9. Render Chat History & Handle New User Input
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📑 View Retrieved Source Paragraphs & Page Numbers"):
                for s in msg["sources"]:
                    st.markdown(f"**📄 Page {s['page']}** *(Cosine Similarity: `{s['score']:.3f}`)*")
                    st.caption(s["text"])
                    st.divider()

user_query = st.chat_input("Ask a question about the loaded PDF document...")
active_query = preset_query if preset_query else user_query

if active_query:
    st.session_state["messages"].append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching FAISS vector index and extracting cited answer..."):
            answer, sources = answer_question(active_query)
            st.markdown(answer)
            with st.expander("📑 View Retrieved Source Paragraphs & Page Numbers", expanded=True):
                for s in sources:
                    st.markdown(f"**📄 Page {s['page']}** *(Cosine Similarity: `{s['score']:.3f}`)*")
                    st.caption(s["text"])
                    st.divider()

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
