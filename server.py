
import os
import json
import re
import logging
from pathlib import Path
from datetime import datetime

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Load .env from the same folder as this file
load_dotenv(BASE_DIR / ".env")

MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

DATABASE_FILE = BASE_DIR / "training_data.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GalaxyNovaAI")

# Load up to 20 Gemini API keys
API_KEYS = []

for number in range(1, 21):
    key = os.getenv(f"GEMINI_API_KEY_{number}")
    if key and key.strip():
        API_KEYS.append(key.strip())

# Also support one general key
if not API_KEYS:
    single_key = os.getenv("GEMINI_API_KEY")
    if single_key and single_key.strip():
        API_KEYS.append(single_key.strip())

logger.info("Galaxy Nova AI starting...")
logger.info("Gemini keys loaded: %s", len(API_KEYS))
logger.info("Gemini model: %s", MODEL)

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="Galaxy Nova AI",
    description="Galaxy Nova AI backend",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# GREETINGS
# --------------------------------------------------

GREETINGS = {
    "hi": "Hello! 👋 Welcome to Galaxy Nova AI. How can I help you today?",
    "hello": "Hey there! 🚀 What would you like to know?",
    "hey": "Hey! How can I help you?",
    "hii": "Hii! 😊 Welcome to Galaxy Nova AI!",
    "good morning": "Good morning! ☀️ How can I help you today?",
    "good afternoon": "Good afternoon! What can I do for you?",
    "good evening": "Good evening! 🌙 How may I assist you?",
    "good night": "Good night! Have a wonderful rest. 🌙",
    "how are you": "I'm doing great and ready to help! How are you?",
    "who are you": "I'm Galaxy Nova AI, your AI assistant for questions, learning, ideas, and more.",
    "what is your name": "I'm Galaxy Nova AI! 🚀",
    "thanks": "You're welcome! 😊",
    "thank you": "Happy to help! Ask me anything else.",
    "bye": "Goodbye! See you again on Galaxy Nova AI. 👋"
}

def greeting_reply(message):
    """Return a greeting response if the message is a greeting."""
    cleaned = re.sub(r"[^a-zA-Z\s]", "", message.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    if cleaned in GREETINGS:
        return GREETINGS[cleaned]

    return None

# --------------------------------------------------
# KNOWLEDGE DATABASE
# --------------------------------------------------

def load_database():
    """Load the local JSON knowledge database."""
    if not DATABASE_FILE.exists():
        logger.warning("training_data.json was not found.")
        return []

    try:
        with open(DATABASE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Support either a list or a dictionary containing entries
        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for key in ("entries", "data", "knowledge", "topics"):
                if isinstance(data.get(key), list):
                    return data[key]

        logger.warning("Unrecognized training_data.json structure.")
        return []

    except Exception:
        logger.exception("Could not load training_data.json.")
        return []


DATABASE = load_database()
logger.info("Knowledge database entries loaded: %s", len(DATABASE))


STOP_WORDS = {
    "the", "is", "am", "are", "was", "were", "a", "an",
    "and", "or", "but", "to", "of", "in", "on", "at",
    "for", "with", "what", "who", "when", "where", "why",
    "how", "can", "could", "would", "should", "do", "does",
    "did", "tell", "me", "about", "please", "explain",
    "i", "you", "it", "this", "that", "my", "your"
}


def tokenize(text):
    """Convert text into useful lowercase keywords."""
    words = re.findall(r"[a-zA-Z0-9]+", str(text).lower())

    return {
        word for word in words
        if len(word) > 1 and word not in STOP_WORDS
    }


def search_database(question):
    """
    Search local knowledge before using Gemini.
    Supports entries with question, keywords, topic,
    answer, response, or text fields.
    """
    question_words = tokenize(question)

    if not question_words:
        return None

    best_answer = None
    best_score = 0

    for entry in DATABASE:
        if not isinstance(entry, dict):
            continue

        answer = (
            entry.get("answer")
            or entry.get("response")
            or entry.get("text")
        )

        if not answer:
            continue

        searchable_text = " ".join([
            str(entry.get("question", "")),
            str(entry.get("topic", "")),
            str(entry.get("category", "")),
            str(entry.get("keywords", "")),
            str(entry.get("title", ""))
        ])

        entry_words = tokenize(searchable_text)

        if not entry_words:
            continue

        common_words = question_words.intersection(entry_words)

        # Give more weight to matching words
        score = len(common_words) / max(len(question_words), 1)

        # Extra weight for exact question matches
        saved_question = str(entry.get("question", "")).lower().strip()
        if saved_question and saved_question == question.lower().strip():
            score += 1.0

        if score > best_score:
            best_score = score
            best_answer = str(answer)

    # Only use a database result when there is a reasonable match
    if best_score >= 0.45:
        logger.info("Answer found in local database.")
        return best_answer

    return None

# --------------------------------------------------
# REQUEST FORMAT
# --------------------------------------------------

class ChatRequest(BaseModel):
    messages: list = []
    message: str = ""


# --------------------------------------------------
# PREPARE GEMINI CONVERSATION
# --------------------------------------------------

SYSTEM_INSTRUCTION = """
You are Galaxy Nova AI, a helpful, friendly and intelligent AI assistant.

Your rules:
1. Answer clearly and accurately.
2. Be friendly and respectful.
3. Explain difficult topics in beginner-friendly language.
4. Use headings and lists when useful.
5. If you do not know something, say so honestly.
6. Never claim that you searched the internet unless you actually did.
7. Do not invent facts, sources, or statistics.
8. Help users with science, coding, mathematics, education,
   creativity, general knowledge, and everyday questions.
"""


def prepare_contents(messages, fallback_message=""):
    """Convert frontend messages into Gemini conversation format."""
    contents = []

    if not messages and fallback_message:
        messages = [{"role": "user", "content": fallback_message}]

    # Keep recent messages to avoid unnecessarily large requests
    for item in messages[-20:]:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "user")).lower()

        if role in ("assistant", "model"):
            gemini_role = "model"
        elif role == "user":
            gemini_role = "user"
        else:
            continue

        content = (
            item.get("content")
            or item.get("text")
            or item.get("parts")
            or ""
        )

        if isinstance(content, list):
            text_parts = []

            for part in content:
                if isinstance(part, dict):
                    text_parts.append(str(part.get("text", "")))
                else:
                    text_parts.append(str(part))

            content = "\n".join(text_parts)

        if not isinstance(content, str):
            content = str(content)

        content = content.strip()

        if content:
            contents.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })

    # Gemini conversation must start with a user message
    while contents and contents[0]["role"] != "user":
        contents.pop(0)

    return contents

# --------------------------------------------------
# GEMINI API WITH MULTI-KEY FALLBACK
# --------------------------------------------------

async def ask_gemini(messages):
    """Try Gemini keys one by one until one succeeds."""

    if not API_KEYS:
        return (
            "No Gemini API keys are configured. "
            "Please add your keys to the environment variables."
        )

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{MODEL}:generateContent"
    )

    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "contents": messages,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048
        }
    }

    errors = []

    async with httpx.AsyncClient(timeout=60.0) as client:

        for index, api_key in enumerate(API_KEYS, start=1):

            try:
                response = await client.post(
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": api_key
                    },
                    json=payload
                )

                if response.status_code == 200:
                    data = response.json()

                    candidates = data.get("candidates", [])

                    if candidates:
                        parts = (
                            candidates[0]
                            .get("content", {})
                            .get("parts", [])
                        )

                        reply = "".join(
                            part.get("text", "")
                            for part in parts
                            if isinstance(part, dict)
                        ).strip()

                        if reply:
                            logger.info(
                                "Gemini response received using key %s.",
                                index
                            )
                            return reply

                    errors.append(f"Key {index}: Empty Gemini response.")
                    continue

                error_text = response.text[:500]

                logger.warning(
                    "Gemini key %s failed with HTTP %s.",
                    index,
                    response.status_code
                )

                errors.append(
                    f"Key {index}: HTTP {response.status_code}: {error_text}"
                )

                # Try the next key if this key fails

            except httpx.TimeoutException:
                errors.append(f"Key {index}: Request timed out.")

            except Exception as error:
                logger.exception("Gemini request failed.")
                errors.append(f"Key {index}: {str(error)}")

    # Give a clearer error for model problems
    if any("404" in error for error in errors):
        return (
            f"Gemini model '{MODEL}' was not found or is unavailable "
            "for your API project. Check the model name and API access "
            "in Google AI Studio."
        )

    if any("429" in error for error in errors):
        return (
            "All available Gemini keys may have reached their usage limits. "
            "Please try again later or check your API quotas."
        )

    logger.error("All Gemini keys failed: %s", errors)

    return (
        "Gemini is temporarily unavailable. Please try again later. "
        "Check your API keys, model access, and Render logs if this continues."
    )

# --------------------------------------------------
# CHAT ROUTE
# --------------------------------------------------

@app.post("/api/chat")
async def chat(request: ChatRequest):

    try:
        messages = prepare_contents(
            request.messages,
            request.message
        )

        if not messages:
            return JSONResponse(
                status_code=400,
                content={"reply": "Please enter a message first."}
            )

        # Get the latest user message
        latest_question = ""

        for item in reversed(messages):
            if item["role"] == "user":
                latest_question = item["parts"][0]["text"]
                break

        if not latest_question:
            return JSONResponse(
                status_code=400,
                content={"reply": "Please send a valid message."}
            )

        # 1. Check greetings
        greeting = greeting_reply(latest_question)

        if greeting:
            return {"reply": greeting, "source": "greeting"}

        # 2. Search local database
        database_answer = search_database(latest_question)

        if database_answer:
            return {
                "reply": database_answer,
                "source": "database"
            }

        # 3. If not found, use Gemini
        gemini_reply = await ask_gemini(messages)

        return {
            "reply": gemini_reply,
            "source": "gemini"
        }

    except Exception:
        logger.exception("Chat route error.")

        return JSONResponse(
            status_code=500,
            content={
                "reply": "An unexpected server error occurred. Please try again."
            }
        )

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "app": "Galaxy Nova AI",
        "model": MODEL,
        "api_keys_loaded": len(API_KEYS),
        "database_entries": len(DATABASE),
        "time": datetime.utcnow().isoformat() + "Z"
    }

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.get("/")
async def home():
    index_file = BASE_DIR / "index.html"

    if index_file.exists():
        return FileResponse(index_file)

    return {
        "app": "Galaxy Nova AI",
        "status": "online",
        "message": "Backend is running. Add index.html to this folder."
    }


# --------------------------------------------------
# START
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000"))
    )
