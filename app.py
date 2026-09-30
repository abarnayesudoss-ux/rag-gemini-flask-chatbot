import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from chatbot_config import CHATBOT_TITLE, SYSTEM_PROMPT
from rag import add_document, retrieve_context

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

@app.route("/")
def home():
    return render_template("index.html", chatbot_title=CHATBOT_TITLE)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"message": "No file selected."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"message": "Please select a file."}), 400

    if client is None:
        return jsonify({"message": "Gemini API key is not configured."}), 500

    try:
        count = add_document(file, client)
        return jsonify({"message": f"Document added successfully. {count} chunks indexed."})
    except ValueError as exc:
        return jsonify({"message": str(exc)}), 400
    except Exception:
        return jsonify({"message": "Could not process the document."}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()

    if not question:
        return jsonify({"reply": "Please enter a question."}), 400

    if client is None:
        return jsonify({"reply": "Gemini API key is not configured."}), 500

    try:
        context = retrieve_context(question, client)
        prompt = f"""{SYSTEM_PROMPT}

Use the retrieved document context below when it is relevant.
Answer only questions related to {CHATBOT_TITLE}.
If the answer is not supported by the retrieved context, say that the uploaded
documents do not contain enough information and avoid inventing facts.

RETRIEVED CONTEXT:
{context if context else "No relevant document context was found."}

USER QUESTION:
{question}
"""
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )
        return jsonify({"reply": response.text or "No response generated."})
    except Exception:
        return jsonify({"reply": "Sorry, I could not process your request right now."}), 500

if __name__ == "__main__":
    app.run(debug=True)
