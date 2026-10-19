"""Interactive question answering over the indexed thesis.  Usage: python ask.py"""
from graph import build_graph


def main():
    app = build_graph()
    print("Ask about the thesis (empty line to quit).")
    while True:
        q = input("\n> ").strip()
        if not q:
            break
        print(app.invoke({"question": q})["answer"])


if __name__ == "__main__":
    main()
