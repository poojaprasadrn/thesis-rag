"""Ingest a PDF into a searchable index.

Usage: python ingest.py path/to/thesis.pdf
Creates index.pkl (chunks + TF-IDF matrix). Runs fully offline.
"""
import pickle
import re
import sys

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

CHUNK_WORDS = 120
OVERLAP_WORDS = 30


def read_pages(path):
    reader = PdfReader(path)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_pages(pages):
    """Split each page into overlapping word windows and keep the page number."""
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for page_no, text in pages:
        words = re.sub(r"\s+", " ", text).split()
        for start in range(0, max(len(words), 1), step):
            piece = words[start:start + CHUNK_WORDS]
            if len(piece) < 25:
                continue
            chunks.append({"page": page_no, "text": " ".join(piece)})
    return chunks


def build_index(chunks):
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1
    )
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    return vectorizer, matrix


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python ingest.py thesis.pdf")
    chunks = chunk_pages(read_pages(sys.argv[1]))
    vectorizer, matrix = build_index(chunks)
    with open("index.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}, f)
    print(f"Indexed {len(chunks)} chunks from {sys.argv[1]}")


if __name__ == "__main__":
    main()
