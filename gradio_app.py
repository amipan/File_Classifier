"""Gradio client that uploads a local PDF to the classifier API."""

import os
from pathlib import Path

import gradio as gr
import httpx

API_URL = os.getenv("CLASSIFIER_API_URL", "http://127.0.0.1:8001")


def classify_local_path(file_path: str | None) -> tuple[str, dict | None]:
    if not file_path or not file_path.strip():
        return "Enter a local PDF file path.", None

    try:
        path = Path(file_path).expanduser().resolve(strict=True)
    except OSError:
        return "File not found. Check the path and try again.", None

    if not path.is_file():
        return "The path does not point to a file.", None
    if path.suffix.lower() != ".pdf":
        return "Only PDF files are supported.", None

    try:
        with path.open("rb") as pdf_file:
            response = httpx.post(
                f"{API_URL}/classify",
                files={"file": (path.name, pdf_file, "application/pdf")},
                timeout=180,
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        try:
            detail = error.response.json().get("detail", error.response.text)
        except ValueError:
            detail = error.response.text
        return f"Classification failed: {detail}", None
    except httpx.HTTPError as error:
        return f"Could not reach classifier API at {API_URL}: {error}", None
    except OSError as error:
        return f"Could not read the PDF: {error}", None

    result = response.json()
    summary = (
        f"{result['category']} | confidence {result['confidence']:.0%} | "
        f"evidence verified: {result['evidence_verified']}"
    )
    return summary, result


def create_page():
    with gr.Blocks(title="Financial Document Classifier") as page:
        gr.Markdown("# Financial document classifier")
        gr.Markdown("Enter a local PDF path. The file contents are uploaded to the classifier API.")
        file_path = gr.Textbox(
            label="Local PDF path",
            placeholder=r"C:\Documents\financial_statement.pdf",
        )
        classify_button = gr.Button("Classify PDF", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        result = gr.JSON(label="Classification result")
        classify_button.click(
            classify_local_path,
            inputs=file_path,
            outputs=[status, result],
        )
    return page


if __name__ == "__main__":
    create_page().queue().launch(server_name="127.0.0.1", server_port=7861)
