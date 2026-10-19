"""Tests for retrieval and routing logic. Run: python test_rag.py  (needs index.pkl)."""
import rag_nodes as n

CASES = [
    # (question, word that should appear in the retrieved text)
    ("What was the inter-annotator agreement of the annotation study?", "alpha"),
    ("Which model performed best for spam classification?", "DistilBERT"),
    ("How was batch inference run on the cluster?", "SLURM"),
    ("Which search engines were analysed?", "Startpage"),
]


def main():
    passed = 0
    for q, must in CASES:
        docs = n.retrieve(n.rewrite_query(q))
        ok = n.grade(docs) and any(must.lower() in d["text"].lower() for d in docs)
        passed += ok
        print(("PASS" if ok else "FAIL"), "-", q, "| top page", docs[0]["page"], "score %.2f" % docs[0]["score"])
    off_topic = n.retrieve("recipe for chocolate cake with butter")
    ok = not n.grade(off_topic)
    passed += ok
    print(("PASS" if ok else "FAIL"), "- off-topic question is rejected, score %.2f" % off_topic[0]["score"])
    print(f"{passed}/{len(CASES) + 1} passed")


if __name__ == "__main__":
    main()
