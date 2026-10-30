# Design decisions — for my own reference before interviews

This is not part of the app. It's a record of the real tradeoffs I made
while building the AI explanation layer, so I can speak to them honestly
in a technical interview instead of inventing an answer on the spot.

## 1. LSTM stays the only thing that predicts. The LLM only explains.

**Decision:** The LLM never sees raw price history and never outputs a
price. It only receives the LSTM's already-computed prediction + a
volatility-derived confidence signal, and its job is to explain that
output and flag when it should be distrusted.

**Alternative I rejected:** Letting the LLM generate or adjust the
prediction itself (e.g. "LSTM says $182, but factor in recent news and
give me a final number").

**Why I rejected it:** An LLM's output is the statistical average of
everything it was trained on — it has no grounding in this specific
stock's actual time-series behavior. Folding it into the prediction
path would mean a plausible-sounding number with less rigor behind it,
not more. Keeping prediction and explanation strictly separate means
every claim in the explanation is traceable back to either the model's
math or a stated heuristic — nothing is invented by the LLM.

## 2. Confidence is volatility-based, not model-uncertainty-based.

**Decision:** Confidence (High/Medium/Low) is computed from the recent
daily-return volatility of the stock, not from the LSTM's own
uncertainty.

**Alternative I rejected:** Proper uncertainty estimation — MC-Dropout
at inference time, or training a small ensemble of LSTMs and using
their disagreement as a confidence signal.

**Why I rejected it:** Both are the "correct" ML answer, but the current
model has no dropout layers and was trained as a single deterministic
network — retraining it with dropout, or training multiple models and
validating the ensemble, wasn't something I could do reliably in the
time available without risking a broken or undertested model going into
the demo. The volatility heuristic is weaker science but it's honest
about what it is, it's fast, and it still produces a genuinely useful
signal (a choppy stock really is harder to predict than a stable one).
This is the tradeoff I'd explain if asked — not hide.

## 3. Dual LLM provider (Groq primary, Gemini fallback) instead of one.

**Decision:** The explanation/chat layer tries Groq first and falls
back to Gemini if that call fails.

**Alternative I rejected:** Just picking one provider and hardcoding it.

**Why:** A live coding/demo environment is exactly where a single
third-party API having a bad moment can tank a demo. The fallback adds
maybe 15 minutes of extra code for real resilience — a small, deliberate
bit of "own the outcome, not just the implementation."

## 4. In-memory context store instead of a database.

**Decision:** The last prediction per symbol is kept in a plain Python
dict (`LAST_CONTEXT`) rather than a database or persistent session store.

**Why:** This is a single-demo-user app built in under two days. A
database would be correct for a real multi-user product but would be
unjustified complexity here — I'd call this out explicitly if asked
"how would you productionize this," rather than pretend the current
setup already handles it.
