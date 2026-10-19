"""LangGraph workflow: rewrite -> retrieve -> grade -> (generate | clarify).

The grading step decides whether the retrieved text is good enough to answer from.
If not, the graph retries once with the original question, then asks for clarification.
"""
from typing import List, TypedDict

from langgraph.graph import END, StateGraph

import rag_nodes as n


class State(TypedDict, total=False):
    question: str
    query: str
    docs: List[dict]
    relevant: bool
    attempts: int
    answer: str


def rewrite_node(state: State) -> State:
    attempts = state.get("attempts", 0)
    query = n.rewrite_query(state["question"]) if attempts == 0 else state["question"]
    return {"query": query, "attempts": attempts + 1}


def retrieve_node(state: State) -> State:
    return {"docs": n.retrieve(state["query"])}


def grade_node(state: State) -> State:
    return {"relevant": n.grade(state["docs"])}


def generate_node(state: State) -> State:
    answer = n.generate(state["question"], state["docs"])
    pages = ", ".join(str(p) for p in sorted({d["page"] for d in state["docs"]}))
    return {"answer": f"{answer}\n\nSources: pages {pages}"}


def clarify_node(state: State) -> State:
    return {"answer": n.CLARIFY}


def route_after_grade(state: State) -> str:
    if state["relevant"]:
        return "generate"
    return "rewrite" if state["attempts"] < 2 else "clarify"


def build_graph():
    g = StateGraph(State)
    g.add_node("rewrite", rewrite_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("generate", generate_node)
    g.add_node("clarify", clarify_node)
    g.set_entry_point("rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite", "clarify": "clarify"})
    g.add_edge("generate", END)
    g.add_edge("clarify", END)
    return g.compile()
