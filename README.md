# 11-711 Assignment 2: End-to-end NLP System Building
**Andrew ID:** bzhang3

## Requirements

```bash
conda create -n anlp2 python=3.11
conda activate anlp2
pip install requests beautifulsoup4 pdfplumber bm25s faiss-cpu sentence-transformers transformers torch
```

## Steps to Reproduce

### 1. Data Collection
```bash
python src/scraper.py
```
Crawls HTML pages and PDFs into `data/raw/` (57 documents).

### 2. Preprocessing
```bash
python src/preprocessor.py
```
Chunks documents into `data/processed/chunks.jsonl` using sentence-aware chunking (default).

### 3. Build Index
```bash
python src/retriever.py --build
```
Builds BM25 and FAISS indexes into `data/index/`.

### 4. Run on Test Set
```bash
python src/run_leaderboard.py
```
Generates `system_outputs/system_output_1.json` using Hybrid (RRF) + Qwen2.5-3B (S4 configuration).

> **Note:** Step 4 requires a GPU (tested on Google Colab T4). Steps 1–3 can run on CPU (tested on MacBook Air M3).

## System Output
`system_outputs/system_output_1.json` — main submission (S4: Hybrid RRF + Qwen2.5-3B-Instruct)
