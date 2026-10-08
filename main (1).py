from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import ask
from tool_logic import get_listings

app = FastAPI(title="DHA Phase 8 Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def chat_page():
    return FileResponse("chat.html")


@app.get("/health")
def health():
    return {"status": "ok", "try": "/ask?q=sab se sasta commercial plot"}


@app.get("/listings/count")
def listings_count():
    try:
        return {"unique_listings": len(get_listings())}
    except Exception as e:
        raise HTTPException(502, f"Could not reach DHA API: {e}")


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "question is empty")
    try:
        return ask(req.question)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/ask")
def ask_get(q: str):
    """Browser se test karne ke liye: /ask?q=..."""
    return chat(ChatRequest(question=q))
