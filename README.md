# File Classifier

A small financial PDF classifier with three parts:

- `classifier_api.py`: FastAPI upload endpoint. It classifies the uploaded PDF content, validates evidence quotes against extracted page text, and returns category, confidence, rationale, and evidence verification details.
- `gradio_app.py`: local Gradio page that accepts a PDF path, checks that the file exists, and uploads its bytes to FastAPI.
- `langgraph_workflow.py`: a one-node LangGraph workflow that sends a hardcoded PDF path to the same API.

## Setup

From this directory, install the dependencies into the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Add your Google GenAI API key to this directory's ignored `.env` file using `GOOGLE_API_KEY` or `GEMINI_API_KEY`. If both are set, the Google GenAI SDK prioritizes `GOOGLE_API_KEY`.

Scanned PDFs use Tesseract OCR. Install Tesseract and ensure it is on `PATH`, or set `TESSERACT_CMD` in `.env` to the executable location.

## Run

Start the API in one terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn classifier_api:app --host 127.0.0.1 --port 8001
```

Start the Gradio page in a second terminal:

```powershell
.\.venv\Scripts\python.exe gradio_app.py
```

Open <http://127.0.0.1:7861>. The API documentation is available at <http://127.0.0.1:8001/docs>.

Enter a PDF path that is readable by the machine running Gradio. The page uploads the PDF bytes; the API does not receive or access the client-side path. The endpoint accepts PDFs up to 25 MB at `POST /classify`.

## LangGraph Example

`langgraph_workflow.py` contains a hardcoded `PDF_PATH` pointing to a sample financial statement in the sibling `GAdkSample` folder. Update `PDF_PATH` for another local PDF, start the FastAPI service, then run:

```powershell
.\.venv\Scripts\python.exe langgraph_workflow.py
```

The workflow sends the file's contents to the same `POST /classify` endpoint and prints the JSON classification result.
