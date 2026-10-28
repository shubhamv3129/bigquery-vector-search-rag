# 🔍 Enterprise GCP RAG Demo — Cloud Storage, Vertex AI & BigQuery Vector Search

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Vertex AI](https://img.shields.io/badge/Vertex_AI-Embeddings_%26_Gemini-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)
![BigQuery](https://img.shields.io/badge/BigQuery-VECTOR__SEARCH-669DF6?style=for-the-badge&logo=googlebigquery&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

An enterprise-grade **Retrieval-Augmented Generation (RAG)** web application built on **Google Cloud**. It demonstrates an end-to-end unstructured document pipeline: uploading PDFs to **Google Cloud Storage (GCS)**, chunking text with sliding overlap, generating 768-dimensional vectors with **Vertex AI (`text-embedding-004`)**, indexing into **BigQuery**, and executing native SQL `VECTOR_SEARCH` queries grounded with **Gemini (`gemini-2.5-flash`)**.

---

## 📐 Architecture

```mermaid
flowchart LR
    subgraph Ingestion["1. Document Ingestion Pipeline"]
        PDF["PDF Upload\n(Streamlit UI)"] --> GCS["Cloud Storage\n(gs://bucket/pdfs/)"]
        PDF --> Chunker["PyPDF Sliding Chunker\n(500 chars, 100 overlap)"]
        Chunker --> EmbedDoc["Vertex AI text-embedding-004\n(RETRIEVAL_DOCUMENT)"]
        EmbedDoc --> BQLoad["BigQuery Batch Load Job\n(rag_demo.doc_chunks)"]
    end

    subgraph Retrieval["2. Vector Search & Grounded Chat"]
        Query["User Question"] --> EmbedQuery["Vertex AI text-embedding-004\n(RETRIEVAL_QUERY)"]
        EmbedQuery --> BQSearch["BigQuery VECTOR_SEARCH\n(COSINE Distance, top_k=3)"]
        BQLoad -.-> BQSearch
        BQSearch --> Gemini["Vertex AI Gemini 2.5 Flash\nGrounded Answer + Citations"]
    end
```

---

## ✨ Key Features

- **Automated GCP Infrastructure Provisioning**: Automatically verifies and creates the target GCS Bucket, BigQuery Dataset (`rag_demo`), and Vector Table (`doc_chunks` with `ARRAY<FLOAT64>` embeddings) on startup.
- **Zero-Buffer BigQuery Load Jobs**: Uses `load_table_from_file` with newline-delimited JSON (`NDJSON`) rather than streaming inserts so ingested vectors are immediately queryable via `VECTOR_SEARCH`.
- **Task-Specific Asymmetric Embeddings**: Uses `RETRIEVAL_DOCUMENT` during corpus indexing and `RETRIEVAL_QUERY` at search time to maximize retrieval precision.
- **Transparent Pipeline Telemetry**: Displays live chunk previews, 768-dimensional vector previews, target SQL statements, and source similarity scores (`1 - distance`).

---

## 🛠️ Tech Stack

| Component | Service / Library | Purpose |
| :--- | :--- | :--- |
| **UI Framework** | Streamlit | Two-tab ingestion & conversational interface |
| **Object Storage** | Google Cloud Storage (`google-cloud-storage`) | Durable archival storage for raw uploaded PDFs |
| **Embedding Model** | Vertex AI `text-embedding-004` | 768-dimensional semantic vector generation |
| **Vector Database** | Google Cloud BigQuery (`VECTOR_SEARCH`) | Serverless SQL vector storage and cosine similarity search |
| **Generative LLM** | Vertex AI `gemini-2.5-flash` | Context-grounded question answering |
| **PDF Parser** | `pypdf` | Page-level text extraction |

---

## 📁 Project Structure

```text
bigquery-vector-search-rag/
├── main.py                  # Primary application entry point
├── app.py                   # Streamlit runner alias
├── feature_rag_pipeline.py  # GCP provisioning, PDF chunking, BigQuery load, and UI logic
├── .env.example             # Environment variable template
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `GCP_PROJECT_ID` | Yes | — | Google Cloud Project ID |
| `GCP_LOCATION` | No | `us-central1` | Vertex AI region |
| `GCS_BUCKET_NAME` | Yes | `<GCP_PROJECT_ID>-rag-docs` | Cloud Storage bucket for uploaded PDFs |
| `BQ_DATASET` | No | `rag_demo` | BigQuery dataset name |
| `BQ_TABLE` | No | `doc_chunks` | BigQuery table name for document chunks and embeddings |
| `CHUNK_SIZE` | No | `500` | Character window per text chunk |
| `CHUNK_OVERLAP` | No | `100` | Character overlap between consecutive chunks |

---

## 🚀 Quickstart

### 1. Install Dependencies

```bash
git clone https://github.com/shubhamv3129/bigquery-vector-search-rag.git
   cd bigquery-vector-search-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure GCP Environment

```bash
cp .env.example .env
# Edit .env with your GCP_PROJECT_ID and GCS_BUCKET_NAME

gcloud auth application-default login
gcloud services enable storage.googleapis.com aiplatform.googleapis.com bigquery.googleapis.com
```

### 3. Run the Application

```bash
streamlit run main.py
```

Open `http://localhost:8501` in your browser.
