"""Simple LangGraph flow that classifies a hardcoded PDF through FastAPI."""

import json
import os
from pathlib import Path
from typing import NotRequired, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

API_URL = os.getenv("CLASSIFIER_API_URL", "http://127.0.0.1:8001")
PDF_PATH = Path(
    r"C:\Users\amipa\AVidhyaCourse\GAdkSample\borrower_agent\financial_statement_test_set\digital_pdfs\cash_flow_statement_digital.pdf"
)


class ClassificationState(TypedDict):
    classification: NotRequired[dict]


def classify_hardcoded_pdf(_state: ClassificationState) -> dict:
    path = PDF_PATH.expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file at {PDF_PATH}")

    with path.open("rb") as pdf_file:
        response = httpx.post(
            f"{API_URL}/classify",
            files={"file": (path.name, pdf_file, "application/pdf")},
            timeout=180,
        )
    response.raise_for_status()
    classification = response.json()
    print(f"Uploaded PDF: {path.name}")
    print(f"Classifier API URL: {API_URL}/classify")
    print(f"Classifier API status code: {response.status_code}")
    print(f"Classifier category: {classification.get('category')}")
    print("Classifier API response:")
    print(json.dumps(classification, indent=2))
    return {"classification": classification}


builder = StateGraph(ClassificationState)
builder.add_node("classify_pdf", classify_hardcoded_pdf)
builder.add_edge(START, "classify_pdf")
builder.add_edge("classify_pdf", END)
workflow = builder.compile()


if __name__ == "__main__":
    workflow.invoke({})
