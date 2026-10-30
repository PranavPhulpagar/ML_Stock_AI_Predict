from flask import Flask, render_template, request, jsonify, session
import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import joblib
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import ai_layer

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# ====== Get Current Folder ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ====== Define LSTM Model ======
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ====== Load Model and Scaler ======
model = LSTMModel()
model_path = os.path.join(BASE_DIR, "model.pth")
scaler_path = os.path.join(BASE_DIR, "scaler.save")

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
scaler = joblib.load(scaler_path)

# In-memory store of the last prediction context, keyed by symbol.
# Good enough for a single-demo-user app; would move to a real session/DB
# store for multi-user use (see DECISIONS.md).
LAST_CONTEXT = {}


def run_prediction(symbol: str):
    data = yf.download(symbol, period="90d", interval="1d")
    if data.empty:
        return None

    close = data['Close'].values.reshape(-1, 1)
    scaled = scaler.transform(close)
    seq = torch.tensor(scaled[-60:].reshape(1, 60, 1), dtype=torch.float32)
    with torch.no_grad():
        pred = model(seq).item()
    predicted_price = float(scaler.inverse_transform([[pred]])[0][0])
    current_price = float(close[-1][0])

    confidence = ai_layer.compute_confidence(close, predicted_price, current_price)
    explanation = ai_layer.generate_explanation(symbol, current_price, predicted_price, confidence)

    context = {
        "symbol": symbol,
        "current_price": current_price,
        "predicted_price": predicted_price,
        "confidence": confidence,
        "explanation": explanation,
    }
    LAST_CONTEXT[symbol] = context
    return context


# ====== Flask Routes ======
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        symbol = request.form["symbol"].upper().strip()
        try:
            result = run_prediction(symbol)
            if result is None:
                error = "Invalid stock symbol or no data available."
            else:
                session["active_symbol"] = symbol
        except Exception as e:
            error = f"Something went wrong: {e}"
    return render_template("index.html", result=result, error=error)


@app.route("/chat", methods=["POST"])
def chat():
    """Follow-up Q&A about the most recently generated prediction."""
    data = request.get_json(force=True)
    question = (data or {}).get("question", "").strip()
    symbol = session.get("active_symbol")

    if not symbol or symbol not in LAST_CONTEXT:
        return jsonify({"answer": "Run a prediction first, then ask me about it."}), 400
    if not question:
        return jsonify({"answer": "Ask me something about the prediction."}), 400

    ctx = LAST_CONTEXT[symbol]
    answer = ai_layer.answer_followup(
        ctx["symbol"], ctx["current_price"], ctx["predicted_price"], ctx["confidence"], question
    )
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
