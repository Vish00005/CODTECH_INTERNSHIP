# 🤖 AI Text Summarization Tool

> Summarize long articles into concise summaries using state-of-the-art NLP — both **extractive** and **abstractive** approaches supported.

---

## 📋 Project Overview

The **AI Text Summarization Tool** is a Python command-line application that condenses lengthy texts (articles, reports, essays) into concise summaries. It supports two fundamentally different NLP paradigms:

| Approach | Library / Model | How it Works |
|---|---|---|
| **Extractive** | `sumy` (LSA / TextRank) | Selects and returns the most important *existing* sentences from the text |
| **Abstractive** | Hugging Face `facebook/bart-large-cnn` | Generates *new* sentences that paraphrase and condense the original text |

### Key Features

- 📝 Paste text manually **or** load from any `.txt` file
- ⚙️  Choose from **3 summarization algorithms**: LSA, TextRank, or BART
- 📊 Displays **word count statistics** and **compression ratio**
- 💾 Saves summaries to `summary_output.txt` (append mode — history preserved)
- 🔍 **Side-by-side comparison** of extractive vs. abstractive outputs

---

## 🗂️ Project Structure

```
text_summarizer/
├── summarizer.py        ← Main application (all logic here)
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
└── sample_article.txt   ← Ready-to-use test article (~500 words)
```

---

## ⚙️ Installation

### Prerequisites

- Python **3.8 or higher**
- pip

### Steps

```bash
# 1. Clone / navigate to the project folder
cd text_summarizer

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install -r requirements.txt
```

> **Note:** The first run of abstractive summarization will automatically download the `facebook/bart-large-cnn` model weights (~1.6 GB) from Hugging Face Hub. Subsequent runs use the local cache and start instantly.

---

## 🚀 Usage

```bash
python summarizer.py
```

You will be greeted by an interactive menu:

```
============================================================
       AI TEXT SUMMARIZATION TOOL  v1.0
       Powered by sumy & Hugging Face Transformers
============================================================

MAIN MENU
────────────────────────────────────────────────────────────
  [1] Enter text manually
  [2] Load text from a file
  [3] Summarize loaded text
  [4] Compare extractive vs abstractive
  [5] Save summary to file
  [6] Quit
────────────────────────────────────────────────────────────
```

### Typical Workflow

```
1. Select [2] → enter:  sample_article.txt
2. Select [3] → choose a method (e.g., [1] LSA)
3. Review the summary and statistics
4. Select [5] to save the result to summary_output.txt
```

---

## 📖 Example Input & Output

**Input (excerpt from `sample_article.txt`):**

> Artificial Intelligence (AI) has rapidly evolved from a futuristic concept depicted in science fiction into a transformative technology that permeates nearly every aspect of modern life. From the recommendation algorithms that curate our social media feeds … [~500 words total]

**Output (Extractive – LSA, 3 sentences):**

```
────────────────────────────────────────────────
  Method           : Extractive – LSA (3 sentences)
  Original Words   : 512
  Summary Words    : 78
  Compression Ratio: 15.2%
────────────────────────────────────────────────

SUMMARY:

AI-driven automation is dramatically increasing productivity across industries
while raising urgent questions about workforce displacement. Machine learning
models trained on millions of medical records can detect early signs of disease
faster than experienced clinicians. The decisions made in the coming years about
how to develop, deploy, and regulate AI will shape the world for generations.
────────────────────────────────────────────────
```

**Output (Abstractive – BART):**

```
────────────────────────────────────────────────
  Method           : Abstractive – BART
  Original Words   : 512
  Summary Words    : 95
  Compression Ratio: 18.6%
────────────────────────────────────────────────

SUMMARY:

Artificial intelligence has evolved from a futuristic concept into a
transformative technology that permeates modern life. AI-driven automation is
increasing productivity but raising questions about job displacement. Healthcare
and education are among the most impacted sectors, with AI tools detecting
disease and personalizing learning. Governance, ethics, and international
cooperation will be critical in shaping AI's future.
────────────────────────────────────────────────
```

---

## 🔧 Function Reference

| Function | Description |
|---|---|
| `extractive_summary(text, sentences_count, algorithm)` | Returns N most important sentences using LSA or TextRank |
| `abstractive_summary(text, max_length, min_length)` | Generates a paraphrased summary using BART |
| `word_count(text)` | Returns the number of words in a string |
| `load_text_from_file(filepath)` | Reads and validates a plain-text file |
| `save_summary(original, summary, method)` | Appends summary + stats to `summary_output.txt` |
| `compare_summaries(text)` | Runs and prints both extractive and abstractive summaries |

---

## 🌱 Future Enhancements

- [ ] **GUI with Tkinter** — A graphical interface with a text area, dropdown for method, and a "Summarize" button
- [ ] **URL support via `newspaper3k`** — Paste a news article URL and automatically extract the text
- [ ] **Multilingual support** — Extend extractive summarization to non-English documents using multilingual models
- [ ] **API endpoint** — Expose the summarizer as a REST API using FastAPI or Flask
- [ ] **PDF / DOCX input** — Support uploading documents directly
- [ ] **Custom models** — Allow users to specify any Hugging Face summarization model

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `transformers` | Hugging Face Transformers (BART model) |
| `torch` | PyTorch backend for the transformer model |
| `sumy` | Extractive summarization (LSA, TextRank) |
| `nltk` | Tokenization and stopword lists |

---

## 📄 License

This project was built as part of the CODETECH Internship program. Free to use for educational purposes.
