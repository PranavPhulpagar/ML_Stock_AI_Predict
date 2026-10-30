"""
ai_layer.py

This module is the "reasoning layer" on top of the raw LSTM prediction.
The LSTM stays the quantitative core (it's what's actually grounded in
time-series signal). The LLM is used ONLY for two things:
  1. Explaining the prediction in plain language, using the model's
     output + a volatility-based confidence signal as context.
  2. Answering free-form follow-up questions about that same prediction.

The LLM never generates the price prediction itself. See DECISIONS.md
for why that split matters.
"""

import os
import numpy as np

# ---------------------------------------------------------------------
# Provider setup — Groq primary, Gemini fallback.
# Both keys are read from environment variables so nothing is hardcoded.
# ---------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

_groq_client = None
_gemini_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None and GROQ_API_KEY:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY, timeout=15.0)
    return _groq_client


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None and GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_client = genai.GenerativeModel("gemini-3.8-flash")
    return _gemini_client


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Tries Groq first (fast, cheap, good for a live demo).
    Falls back to Gemini if Groq fails or no Groq key is set.
    Raises a clear error if neither provider is available/working.
    """
    groq_client = _get_groq_client()
    if groq_client is not None:
        try:
            resp = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=300,
                timeout=15.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ai_layer] Groq call failed, falling back to Gemini: {e}")

    gemini_client = _get_gemini_client()
    if gemini_client is not None:
        try:
            resp = gemini_client.generate_content(
                f"{system_prompt}\n\n{user_prompt}",
                request_options={"timeout": 15},
            )
            return resp.text.strip()
        except Exception as e:
            print(f"[ai_layer] Gemini call failed: {e}")

    return (
        "⚠️ No AI provider is reachable right now (check GROQ_API_KEY / "
        "GEMINI_API_KEY). Showing the raw model output only."
    )


# ---------------------------------------------------------------------
# Confidence signal
#
# The LSTM has no built-in uncertainty estimate (no dropout at inference,
# no ensemble). Rather than bolt on MC-Dropout or a bootstrap ensemble
# (accurate, but too slow to retrain/validate in the time we have),
# we use a volatility-based heuristic: a stock that's been choppy over
# the last ~30 days gets a wider error band and a lower confidence
# label than one that's been stable. This is a DELIBERATE simplification
# — documented as a rejected alternative in DECISIONS.md — not an
# oversight.
# ---------------------------------------------------------------------
def compute_confidence(close_prices: np.ndarray, predicted_price: float, current_price: float):
    """
    close_prices: 1D array of recent closing prices (most recent last)
    Returns a dict with a confidence label, a +/- price band, and the
    raw volatility number so the explanation layer can reference it.
    """
    prices = close_prices.flatten()
    recent = prices[-30:] if len(prices) >= 30 else prices

    daily_returns = np.diff(recent) / recent[:-1]
    volatility = float(np.std(daily_returns))  # daily return std dev

    # Thresholds are heuristic, tuned loosely against typical large-cap
    # daily volatility (~1-2%) vs. high-volatility names (4%+).
    if volatility < 0.015:
        label = "High"
    elif volatility < 0.035:
        label = "Medium"
    else:
        label = "Low"

    band = predicted_price * volatility * 1.5  # rough +/- band
    pct_move = ((predicted_price - current_price) / current_price) * 100

    return {
        "label": label,
        "volatility": round(volatility * 100, 2),  # as %
        "band": round(band, 2),
        "pct_move": round(pct_move, 2),
    }


# ---------------------------------------------------------------------
# Explanation generation
# ---------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a markets explanation assistant sitting on top of an LSTM "
    "next-day stock price model. You do NOT generate predictions yourself "
    "— you only explain a prediction that has already been computed, and "
    "you are honest about the model's limits. Keep responses tight: 3-5 "
    "sentences for explanations, 2-4 for follow-up answers. Never claim "
    "certainty the model doesn't have. If confidence is Low, say so plainly "
    "and explain why (recent volatility) rather than hedging vaguely."
)


def generate_explanation(symbol: str, current_price: float, predicted_price: float, confidence: dict) -> str:
    pct = confidence["pct_move"]
    direction = "up" if pct >= 0 else "down"
    user_prompt = (
        f"Stock: {symbol}\n"
        f"Current price: ${current_price:.2f}\n"
        f"LSTM next-day prediction: ${predicted_price:.2f} ({direction} {abs(pct)}%)\n"
        f"Confidence: {confidence['label']} (recent daily volatility: {confidence['volatility']}%)\n"
        f"Estimated error band: +/- ${confidence['band']}\n\n"
        "Explain this prediction in plain language for someone deciding "
        "whether to trust it. Mention the confidence level and why, based "
        "on the volatility given. Do not restate the raw numbers mechanically "
        "— interpret them."
    )
    return _call_llm(SYSTEM_PROMPT, user_prompt)


def answer_followup(symbol: str, current_price: float, predicted_price: float, confidence: dict, question: str) -> str:
    pct = confidence["pct_move"]
    context = (
        f"Context — Stock: {symbol}, current price: ${current_price:.2f}, "
        f"predicted next-day price: ${predicted_price:.2f} ({pct}% move), "
        f"confidence: {confidence['label']} (volatility {confidence['volatility']}%), "
        f"error band: +/- ${confidence['band']}.\n\n"
        f"User question: {question}"
    )
    return _call_llm(SYSTEM_PROMPT, context)
