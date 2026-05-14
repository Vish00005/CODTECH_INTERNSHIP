"""
AI Text Summarization Tool
==========================
Author  : CODETECH Intern
Date    : 2026
Version : 1.0

This module provides two summarization strategies:
  1. Extractive  – uses the `sumy` library (LSA algorithm by default)
  2. Abstractive – uses Hugging Face Transformers (facebook/bart-large-cnn)

Supports:
  - Manual multi-line text input
  - Loading text from a file
  - Saving the generated summary to summary_output.txt
  - Displaying word-count statistics and compression ratio
"""

# ──────────────────────────────────────────────────────────────────
# Standard-library imports
# ──────────────────────────────────────────────────────────────────
import os
import sys
import textwrap
import datetime

# ──────────────────────────────────────────────────────────────────
# Third-party imports (installed via requirements.txt)
# ──────────────────────────────────────────────────────────────────
import nltk

# sumy – extractive summarization
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Hugging Face Transformers – abstractive summarization
from transformers import pipeline

# ──────────────────────────────────────────────────────────────────
# NLTK data – download quietly if not already present
# ──────────────────────────────────────────────────────────────────
def _ensure_nltk_data() -> None:
    """Download required NLTK corpora / models if they are missing."""
    resources = [
        ("tokenizers/punkt",        "punkt"),
        ("tokenizers/punkt_tab",    "punkt_tab"),
        ("corpora/stopwords",       "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"  [NLTK] Downloading '{name}' … ", end="", flush=True)
            nltk.download(name, quiet=True)
            print("done.")


# ──────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────
def word_count(text: str) -> int:
    """Return the number of whitespace-separated words in *text*."""
    return len(text.split())


def _wrap(text: str, width: int = 88) -> str:
    """Wrap text to *width* columns for clean terminal display."""
    return textwrap.fill(text, width=width)


def _divider(char: str = "─", width: int = 60) -> str:
    return char * width


# ──────────────────────────────────────────────────────────────────
# Extractive summarization  (sumy / LSA or TextRank)
# ──────────────────────────────────────────────────────────────────
def extractive_summary(
    text: str,
    sentences_count: int = 3,
    algorithm: str = "lsa",
    language: str = "english",
) -> str:
    """
    Produce an extractive summary by selecting the most important
    sentences from *text*.

    Parameters
    ----------
    text            : The input article / document.
    sentences_count : Number of sentences to include in the summary.
    algorithm       : 'lsa'  (Latent Semantic Analysis, default)
                      'textrank' (graph-based ranking).
    language        : Language of the document (default 'english').

    Returns
    -------
    A string containing the selected sentences joined by spaces.
    """
    if not text.strip():
        raise ValueError("Input text is empty.")

    # Parse the plain text
    parser   = PlaintextParser.from_string(text, Tokenizer(language))
    stemmer  = Stemmer(language)

    # Choose the algorithm
    algo = algorithm.lower().strip()
    if algo == "lsa":
        summarizer = LsaSummarizer(stemmer)
    elif algo == "textrank":
        summarizer = TextRankSummarizer(stemmer)
    else:
        raise ValueError(f"Unknown extractive algorithm: '{algorithm}'. "
                         "Choose 'lsa' or 'textrank'.")

    summarizer.stop_words = get_stop_words(language)

    # Generate summary sentences
    summary_sentences = summarizer(parser.document, sentences_count)

    if not summary_sentences:
        return "Could not extract a summary from the provided text."

    return " ".join(str(s) for s in summary_sentences)


# ──────────────────────────────────────────────────────────────────
# Abstractive summarization  (Hugging Face / BART)
# ──────────────────────────────────────────────────────────────────
# Cache the pipeline so it is only loaded once per session.
_ABSTRACTIVE_PIPELINE = None
_ABSTRACTIVE_MODEL    = "facebook/bart-large-cnn"


def abstractive_summary(
    text: str,
    max_length: int = 150,
    min_length: int = 40,
) -> str:
    """
    Generate an abstractive summary using a pre-trained sequence-to-sequence
    transformer (facebook/bart-large-cnn by default).

    Parameters
    ----------
    text       : The input article / document.
    max_length : Maximum number of tokens in the generated summary.
    min_length : Minimum number of tokens in the generated summary.

    Returns
    -------
    A string containing the generated summary.

    Notes
    -----
    * The first call downloads the model weights (~1.6 GB) from
      Hugging Face Hub and caches them locally.
    * Subsequent calls reuse the cached pipeline.
    """
    global _ABSTRACTIVE_PIPELINE

    if not text.strip():
        raise ValueError("Input text is empty.")

    # BART has a token limit (~1 024 tokens / ~800 words).
    # We truncate silently and warn the user.
    MAX_WORDS = 800
    words = text.split()
    if len(words) > MAX_WORDS:
        print(f"\n  [INFO] Text truncated to {MAX_WORDS} words for BART "
              "(model token-limit).")
        text = " ".join(words[:MAX_WORDS])

    # Load the pipeline lazily
    if _ABSTRACTIVE_PIPELINE is None:
        print(f"\n  [INFO] Loading model '{_ABSTRACTIVE_MODEL}' …")
        print("  (First run downloads ~1.6 GB; subsequent runs use cache.)\n")
        _ABSTRACTIVE_PIPELINE = pipeline(
            "summarization",
            model=_ABSTRACTIVE_MODEL,
        )
        print("  [INFO] Model loaded.\n")

    result = _ABSTRACTIVE_PIPELINE(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )

    return result[0]["summary_text"]


# ──────────────────────────────────────────────────────────────────
# File I/O helpers
# ──────────────────────────────────────────────────────────────────
def load_text_from_file(filepath: str) -> str:
    """
    Read and return the contents of a plain-text file.

    Raises
    ------
    FileNotFoundError if the path does not exist.
    ValueError        if the file is empty.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: '{filepath}'")

    with open(filepath, "r", encoding="utf-8") as fh:
        content = fh.read()

    if not content.strip():
        raise ValueError(f"File '{filepath}' is empty.")

    return content


def save_summary(
    original_text: str,
    summary: str,
    method: str,
    output_path: str = "summary_output.txt",
) -> None:
    """
    Append the summary and its statistics to *output_path*.

    The file is opened in append mode so multiple sessions are preserved.
    """
    orig_wc    = word_count(original_text)
    summ_wc    = word_count(summary)
    ratio      = (summ_wc / orig_wc * 100) if orig_wc else 0.0
    timestamp  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    separator  = "=" * 60

    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write(f"{separator}\n")
        fh.write(f"Timestamp        : {timestamp}\n")
        fh.write(f"Method           : {method}\n")
        fh.write(f"Original Words   : {orig_wc}\n")
        fh.write(f"Summary Words    : {summ_wc}\n")
        fh.write(f"Compression Ratio: {ratio:.1f}%\n")
        fh.write(f"{separator}\n\n")
        fh.write("SUMMARY:\n")
        fh.write(textwrap.fill(summary, width=80))
        fh.write("\n\n")

    print(f"\n  ✔  Summary saved to '{os.path.abspath(output_path)}'")


# ──────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────
def _print_stats(original: str, summary: str, method: str) -> None:
    """Print word-count statistics and the summary to stdout."""
    orig_wc = word_count(original)
    summ_wc = word_count(summary)
    ratio   = (summ_wc / orig_wc * 100) if orig_wc else 0.0

    print(f"\n{_divider()}")
    print(f"  Method           : {method}")
    print(f"  Original Words   : {orig_wc}")
    print(f"  Summary Words    : {summ_wc}")
    print(f"  Compression Ratio: {ratio:.1f}%")
    print(f"{_divider()}")
    print("\nSUMMARY:\n")
    print(_wrap(summary))
    print(f"{_divider()}\n")


# ──────────────────────────────────────────────────────────────────
# Interactive menu helpers
# ──────────────────────────────────────────────────────────────────
def _get_text_from_user() -> str:
    """
    Prompt the user to enter multi-line text.
    They finish input by typing 'END' on its own line (or Ctrl-D / Ctrl-Z).
    """
    print("\nPaste your article below.")
    print("When finished, type  END  on a new line and press Enter.\n")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)

    text = "\n".join(lines).strip()
    if not text:
        raise ValueError("No text was entered.")
    return text


def _get_text_from_file_prompt() -> str:
    """Prompt the user for a file path and load the text."""
    path = input("\nEnter the file path: ").strip()
    return load_text_from_file(path)


def _choose_method() -> str:
    """
    Ask the user to choose a summarization method.
    Returns one of: 'extractive_lsa', 'extractive_textrank', 'abstractive'.
    """
    print("\nSummarization Method:")
    print("  [1] Extractive – LSA")
    print("  [2] Extractive – TextRank")
    print("  [3] Abstractive – BART (Hugging Face)")

    while True:
        choice = input("\nEnter choice (1/2/3): ").strip()
        if choice == "1":
            return "extractive_lsa"
        elif choice == "2":
            return "extractive_textrank"
        elif choice == "3":
            return "abstractive"
        else:
            print("  Invalid choice. Please enter 1, 2, or 3.")


def _get_sentence_count() -> int:
    """Ask for the number of sentences for extractive summarization."""
    while True:
        raw = input(
            "Number of sentences for extractive summary [default 3]: "
        ).strip()
        if raw == "":
            return 3
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("  Please enter a positive integer.")


# ──────────────────────────────────────────────────────────────────
# Side-by-side comparison (optional enhancement)
# ──────────────────────────────────────────────────────────────────
def compare_summaries(text: str, sentences_count: int = 3) -> None:
    """
    Generate and display both extractive (LSA) and abstractive summaries
    side by side for comparison.
    """
    print(f"\n{_divider('═')}")
    print("   EXTRACTIVE SUMMARY (LSA)")
    print(_divider('═'))
    ext = extractive_summary(text, sentences_count=sentences_count)
    print(_wrap(ext))

    print(f"\n{_divider('═')}")
    print("   ABSTRACTIVE SUMMARY (BART)")
    print(_divider('═'))
    abst = abstractive_summary(text)
    print(_wrap(abst))
    print(_divider('═'))


# ──────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Interactive command-line menu for the AI Text Summarization Tool.
    """
    _ensure_nltk_data()

    # ── Banner ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("       AI TEXT SUMMARIZATION TOOL  v1.0")
    print("       Powered by sumy & Hugging Face Transformers")
    print("=" * 60)

    text   = ""
    summary = ""
    method  = ""

    while True:
        # ── Main Menu ─────────────────────────────────────────────
        print("\nMAIN MENU")
        print(_divider())
        print("  [1] Enter text manually")
        print("  [2] Load text from a file")
        print("  [3] Summarize loaded text")
        print("  [4] Compare extractive vs abstractive")
        print("  [5] Save summary to file")
        print("  [6] Quit")
        print(_divider())

        choice = input("Your choice: ").strip()

        # ── 1. Manual text input ──────────────────────────────────
        if choice == "1":
            try:
                text = _get_text_from_user()
                print(f"\n  ✔  Text loaded ({word_count(text)} words).")
            except ValueError as exc:
                print(f"\n  ✘  {exc}")

        # ── 2. Load from file ─────────────────────────────────────
        elif choice == "2":
            try:
                text = _get_text_from_file_prompt()
                print(f"\n  ✔  File loaded ({word_count(text)} words).")
            except (FileNotFoundError, ValueError) as exc:
                print(f"\n  ✘  {exc}")

        # ── 3. Summarize ──────────────────────────────────────────
        elif choice == "3":
            if not text:
                print("\n  ✘  No text loaded. Use option 1 or 2 first.")
                continue

            try:
                method = _choose_method()

                if method == "extractive_lsa":
                    n = _get_sentence_count()
                    print("\n  Generating extractive (LSA) summary …")
                    summary = extractive_summary(
                        text, sentences_count=n, algorithm="lsa"
                    )
                    label = f"Extractive – LSA ({n} sentences)"

                elif method == "extractive_textrank":
                    n = _get_sentence_count()
                    print("\n  Generating extractive (TextRank) summary …")
                    summary = extractive_summary(
                        text, sentences_count=n, algorithm="textrank"
                    )
                    label = f"Extractive – TextRank ({n} sentences)"

                else:  # abstractive
                    max_len = input(
                        "Max summary tokens [default 150]: "
                    ).strip()
                    min_len = input(
                        "Min summary tokens [default  40]: "
                    ).strip()
                    max_len = int(max_len) if max_len.isdigit() else 150
                    min_len = int(min_len) if min_len.isdigit() else 40
                    print("\n  Generating abstractive (BART) summary …")
                    summary = abstractive_summary(
                        text, max_length=max_len, min_length=min_len
                    )
                    label = "Abstractive – BART"

                _print_stats(text, summary, label)

            except (ValueError, Exception) as exc:
                print(f"\n  ✘  Error during summarization: {exc}")

        # ── 4. Compare both methods ───────────────────────────────
        elif choice == "4":
            if not text:
                print("\n  ✘  No text loaded. Use option 1 or 2 first.")
                continue
            try:
                compare_summaries(text)
            except Exception as exc:
                print(f"\n  ✘  Error: {exc}")

        # ── 5. Save summary ───────────────────────────────────────
        elif choice == "5":
            if not summary:
                print("\n  ✘  No summary generated yet. Use option 3 first.")
                continue
            try:
                save_summary(text, summary, method or "unknown")
            except OSError as exc:
                print(f"\n  ✘  Could not save file: {exc}")

        # ── 6. Quit ───────────────────────────────────────────────
        elif choice == "6":
            print("\n  Goodbye!\n")
            sys.exit(0)

        else:
            print("\n  ✘  Invalid choice. Please enter a number 1–6.")


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
