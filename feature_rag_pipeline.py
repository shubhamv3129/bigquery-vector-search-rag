"""
Feature module for the Enterprise GCP RAG Pipeline Streamlit application.
Orchestrates PDF ingestion to Google Cloud Storage, text extraction and chunking,
Vertex AI vector embedding generation, BigQuery vector storage, and VECTOR_SEARCH Q&A.
"""
import hashlib
import io
import json
import os
import re
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from google.cloud import bigquery, storage
from pypdf import PdfReader
import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", f"{PROJECT_ID}-rag-docs")
BQ_DATASET = os.getenv("BQ_DATASET", "rag_demo")
BQ_TABLE = os.getenv("BQ_TABLE", "doc_chunks")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


def get_clients() -> Dict[str, Any]:
    """
    Initializes and returns Google Cloud Storage, BigQuery, Vertex AI Embedding,
    and Gemini clients.

    Returns:
        Dict[str, Any]: Dictionary of initialized GCP service clients.
    """
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    return {
        "gcs": storage.Client(project=PROJECT_ID),
        "bq": bigquery.Client(project=PROJECT_ID),
        "embed": TextEmbeddingModel.from_pretrained("text-embedding-004"),
        "gemini": GenerativeModel("gemini-2.5-flash"),
    }


def setup_infrastructure(clients: Dict[str, Any]) -> None:
    """
    Ensures the target Cloud Storage bucket and BigQuery dataset/table exist.

    Args:
        clients (Dict[str, Any]): Initialized GCP clients.
    """
    try:
        clients["gcs"].get_bucket(BUCKET_NAME)
    except Exception:
        clients["gcs"].create_bucket(BUCKET_NAME, location="US")

    ds_ref = bigquery.DatasetReference(PROJECT_ID, BQ_DATASET)
    try:
        clients["bq"].get_dataset(ds_ref)
    except Exception:
        ds = bigquery.Dataset(ds_ref)
        ds.location = "US"
        clients["bq"].create_dataset(ds)

    schema = [
        bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("source_file", "STRING"),
        bigquery.SchemaField("text", "STRING"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        bigquery.SchemaField("ingested_at", "TIMESTAMP"),
    ]
    table_ref = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
    try:
        clients["bq"].get_table(table_ref)
    except Exception:
        clients["bq"].create_table(bigquery.Table(table_ref, schema=schema))


def process_pdf(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF byte stream and splits it into overlapping chunks.

    Args:
        file_bytes (bytes): Raw PDF file content.
        filename (str): Original PDF filename.

    Returns:
        List[Dict[str, Any]]: Extracted text chunks with deterministic IDs.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    full_text = " ".join([page.extract_text() or "" for page in reader.pages])
    full_text = re.sub(r"\s+", " ", full_text).strip()

    chunks: List[Dict[str, Any]] = []
    start = 0
    idx = 0
    while start < len(full_text):
        end = min(start + CHUNK_SIZE, len(full_text))
        ctext = full_text[start:end].strip()
        if ctext:
            chunk_hash = hashlib.md5(f"{filename}_{idx}".encode()).hexdigest()[:16]
            chunks.append(
                {
                    "chunk_id": chunk_hash,
                    "source_file": filename,
                    "text": ctext,
                }
            )
            idx += 1
        start += CHUNK_SIZE - OVERLAP
    return chunks


def save_to_bigquery_instantly(
    clients: Dict[str, Any], chunks: List[Dict[str, Any]]
) -> None:
    """
    Loads embedded document chunks into BigQuery using a batch Load Job so
    vectors are immediately available for `VECTOR_SEARCH`.

    Args:
        clients (Dict[str, Any]): Initialized GCP clients.
        chunks (List[Dict[str, Any]]): Embedded chunk dictionaries.
    """
    table_ref = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for chunk in chunks:
        chunk["ingested_at"] = now

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    ndjson_data = "\n".join(json.dumps(c) for c in chunks)
    job = clients["bq"].load_table_from_file(
        io.StringIO(ndjson_data), table_ref, job_config=job_config
    )
    job.result()


def render_rag_demo_app() -> None:
    """
    Renders the Enterprise RAG Pipeline Streamlit user interface.
    """
    st.set_page_config(page_title="GCP RAG Demo", page_icon="🔍", layout="wide")
    st.title("🔍 Enterprise RAG Pipeline on Google Cloud")
    st.markdown(
        "End-to-End Workflow: **PDF Upload → Cloud Storage → Vertex AI Embeddings → BigQuery Vector Search → Gemini**"
    )

    if PROJECT_ID == "your-gcp-project-id":
        st.warning(
            "Please configure `GCP_PROJECT_ID` and `GCS_BUCKET_NAME` in your `.env` file before running queries."
        )
        st.stop()

    clients = get_clients()
    setup_infrastructure(clients)

    tab1, tab2 = st.tabs(["📤 1. Upload & Process Data", "💬 2. Chat with Data"])

    with tab1:
        st.markdown("### Add new documents to your Knowledge Base")
        uploaded_files = st.file_uploader(
            "Drop PDFs here", type=["pdf"], accept_multiple_files=True
        )

        if uploaded_files and st.button("Process Documents", type="primary"):
            for uf in uploaded_files:
                with st.status(f"Processing {uf.name}...", expanded=True) as status:
                    file_bytes = uf.read()

                    st.write("☁️ **Step 1: Uploading to Cloud Storage...**")
                    blob = clients["gcs"].bucket(BUCKET_NAME).blob(f"pdfs/{uf.name}")
                    blob.upload_from_string(file_bytes, content_type="application/pdf")
                    st.caption(f"✓ Saved to `gs://{BUCKET_NAME}/pdfs/{uf.name}`")

                    st.write("✂️ **Step 2: Extracting and chunking text...**")
                    chunks = process_pdf(file_bytes, uf.name)

                    col1, col2 = st.columns(2)
                    col1.metric("File Size", f"{len(file_bytes) / 1024:.1f} KB")
                    col2.metric("Chunks Generated", len(chunks))

                    if chunks:
                        with st.expander("👀 Preview Document Chunk #1"):
                            st.info(chunks[0]["text"])

                    st.write("🧠 **Step 3: Generating vector embeddings...**")
                    progress_bar = st.progress(
                        0, text="Calling Vertex AI text-embedding-004..."
                    )

                    for i in range(0, len(chunks), 250):
                        batch = chunks[i : i + 250]
                        inputs = [
                            TextEmbeddingInput(c["text"], "RETRIEVAL_DOCUMENT")
                            for c in batch
                        ]
                        embeddings = clients["embed"].get_embeddings(inputs)
                        for chunk, emb in zip(batch, embeddings):
                            chunk["embedding"] = emb.values

                        pct = min((i + 250) / max(len(chunks), 1), 1.0)
                        progress_bar.progress(
                            pct,
                            text=f"Embedded {min(i + 250, len(chunks))}/{len(chunks)} chunks",
                        )

                    if chunks:
                        with st.expander(
                            "👀 Preview Mathematical Vector (768 Dimensions)"
                        ):
                            sample_vec = chunks[0]["embedding"][:5]
                            st.code(
                                str([round(v, 4) for v in sample_vec])
                                + " ... [763 more dimensions]"
                            )

                    st.write("🗃️ **Step 4: Saving vectors to BigQuery...**")
                    save_to_bigquery_instantly(clients, chunks)
                    with st.expander("👀 View BigQuery Table Destination"):
                        st.code(
                            f"INSERT INTO `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`",
                            language="sql",
                        )

                    status.update(
                        label=f"Successfully processed {uf.name}", state="complete"
                    )

                st.success(
                    "✅ Pipeline complete! Your data is now indexed in BigQuery and ready for chat."
                )

    with tab2:
        st.markdown("### Ask questions based on your uploaded data")

        question = st.text_input("Ask a question:")
        if st.button("Generate Answer") and question:
            with st.spinner("Searching BigQuery vectors and generating answer..."):
                q_emb = (
                    clients["embed"]
                    .get_embeddings([TextEmbeddingInput(question, "RETRIEVAL_QUERY")])[
                        0
                    ]
                    .values
                )

                emb_json = json.dumps(q_emb)
                sql = f"""
                SELECT
                    base.source_file AS source_file,
                    base.text AS text,
                    distance
                FROM VECTOR_SEARCH(
                    TABLE `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`, 'embedding',
                    (SELECT {emb_json} AS embedding), top_k => 3, distance_type => 'COSINE'
                )
                """
                results = list(clients["bq"].query(sql).result())

                if not results:
                    st.warning("No context found. Please upload documents first.")
                else:
                    context = "\n\n".join(
                        [f"[Source: {r.source_file}]\n{r.text}" for r in results]
                    )
                    prompt = (
                        f"Answer the question using ONLY the context below.\n\n"
                        f"Context:\n{context}\n\nQuestion: {question}"
                    )

                    answer = clients["gemini"].generate_content(prompt).text

                    st.markdown("### Answer")
                    st.info(answer)

                    with st.expander("View Retrieved Sources"):
                        for r in results:
                            st.markdown(
                                f"**{r.source_file}** (Similarity: {round(1 - r.distance, 3)})"
                            )
                            st.caption(r.text)

