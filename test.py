import os
import logging
from flask import Flask, request, jsonify
from app.services.agent import generate_response  # Import the function from agent.py
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Flask endpoint to receive user messages and return property search responses.
    """
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"status": "error", "message": "Invalid request. 'message' field is required."}), 400

        user_message = data["message"]
        logging.info(f"Received message: {user_message}")

        response = generate_response(user_message)

        return jsonify({"status": "ok", "response": response}), 200

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
