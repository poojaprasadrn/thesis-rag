# rag_nodes version 3 (decimal-number bonus in extractive fallback)
"""Plain-Python building blocks used by the LangGraph workflow (and testable without it)."""
import os
import pickle
import re

import numpy as np

TOP_K = 6
MIN_SCORE = 0.08  # below this the retrieved text is treated as not relevant

_index = None


def load_index(path="index.pkl"):
    global _index
    if _index is None:
        with open(path, "rb") as f:
            _index = pickle.load(f)
    return _index


def rewrite_query(question: str) -> str:
    """Light query clean-up: drop filler so retrieval focuses on content words."""
    q = re.sub(r"\b(please|can you|could you|tell me|explain|what is|what are|what's)\b", " ", question, flags=re.I)
    q = re.sub(r"\s+", " ", q).strip(" ?.")
    q = q or question
    # Light query expansion: add the thesis' own terms for common paraphrases.
    expansions = {
        "agreement": "Krippendorff alpha average",
        "annotator": "annotators ratings",
        "best": "performed best accuracy",
        "accuracy": "accuracy F1",
        "cluster": "SLURM Docker nodes",
    }
    extra = [v for k, v in expansions.items() if k in q.lower()]
    return (q + " " + " ".join(extra)).strip()


def retrieve(query: str, k: int = TOP_K):
    idx = load_index()
    qvec = idx["vectorizer"].transform([query])
    scores = (idx["matrix"] @ qvec.T).toarray().ravel()
    top = np.argsort(scores)[::-1][:k]
    return [
        {"page": idx["chunks"][i]["page"], "text": idx["chunks"][i]["text"], "score": float(scores[i])}
        for i in top
    ]


def grade(docs) -> bool:
    """Is the best retrieved chunk relevant enough to answer from?"""
    return bool(docs) and docs[0]["score"] >= MIN_SCORE


def _extractive_answer(question, docs):
    """No-LLM fallback: score every sentence in the retrieved chunks by word overlap with the
    question, with a bonus for sentences that contain numbers (results, metrics)."""
    stop = {"the", "a", "an", "of", "in", "was", "is", "what", "how", "which", "did", "do", "and", "to", "for", "on"}
    qwords = {w for w in re.findall(r"\w+", question.lower()) if w not in stop}
    wants_number = bool(re.search(r"\b(how many|how much|what was|what were|accuracy|score|f1|agreement|alpha)\b", question, re.I))
    scored = []
    for d in docs:
        for sent in re.split(r"(?<=[.!?])\s+", d["text"]):
            words = set(re.findall(r"\w+", sent.lower()))
            overlap = len(qwords & words)
            if not overlap or len(sent) < 25:
                continue
            bonus = (3 if re.search(r"\d\.\d", sent) else 1) if wants_number and re.search(r"\d", sent) else 0
            scored.append((overlap + bonus + d["score"], sent.strip()))
    if not scored:
        return docs[0]["text"][:300]
    scored.sort(key=lambda t: -t[0])
    out, seen = [], set()
    for _, sent in scored:
        if sent not in seen:
            out.append(sent)
            seen.add(sent)
        if len(out) == 2:
            break
    return " ".join(out)


def generate(question, docs) -> str:
    """Answer only from the retrieved context. Uses Claude if ANTHROPIC_API_KEY is set,
    otherwise returns the most relevant sentences (extractive fallback)."""
    context = "\n\n".join(f"[page {d['page']}] {d['text']}" for d in docs)
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=os.environ.get("RAG_MODEL", "claude-sonnet-4-5"),
            max_tokens=500,
            system=(
                "Answer the question using ONLY the context from the thesis. "
                "Cite pages like (p. 12). If the context does not contain the answer, say so."
            ),
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        )
        return msg.content[0].text
    return _extractive_answer(question, docs)


CLARIFY = (
    "I couldn't find that in the thesis. Try asking about the annotation study, the "
    "classifiers, the SLURM batch inference or the trend results."
)
