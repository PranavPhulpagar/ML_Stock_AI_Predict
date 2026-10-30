# 📈 AI-Based Stock Price Prediction System

An AI-powered stock price prediction web application that uses a **Long Short-Term Memory (LSTM)** neural network to predict stock prices and Generative AI to explain predictions in simple language.

## 🚀 Project Overview

This project combines Machine Learning and Generative AI to provide stock price predictions along with AI-generated explanations.

The LSTM model analyzes historical stock closing prices to estimate the next price. An AI explanation layer uses Groq and Google Gemini to explain the prediction, provide a volatility-based confidence indicator, and answer follow-up questions.

**Important:** The LSTM generates the numerical prediction. The LLM only explains the result; it does not generate or modify the predicted price.

## ✨ Features

- Stock symbol-based prediction.
- Historical stock data retrieval using Yahoo Finance.
- LSTM-based time-series prediction using PyTorch.
- Volatility-based High, Medium, and Low confidence indicators.
- AI-generated prediction explanations.
- Follow-up conversational questions about predictions.
- Groq primary provider with Gemini fallback.
- Web interface built using Flask, HTML, CSS, and JavaScript.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Machine Learning | PyTorch, LSTM |
| Data Processing | NumPy, Scikit-learn |
| Stock Data | Yahoo Finance API (`yfinance`) |
| Generative AI | Groq API, Google Gemini API |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Model Storage | PyTorch, Joblib |

## 📁 Project Structure

```text
ML_Stock_AI_Predict/
│
├── app.py
├── ai_layer.py
├── train_model.py
├── model.pth
├── scaler.save
├── requirements.txt
├── .gitignore
├── README.md
│
└── templates/
    └── index.html
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ML_Stock_AI_Predict.git
cd ML_Stock_AI_Predict
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the root project directory:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
FLASK_SECRET_KEY=your_secret_key
```


### 5. Run the application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000/
```

Enter a stock symbol (for example, AAPL) to generate a prediction.

## 🧠 How It Works

1. User enters a stock ticker symbol.
2. The application fetches recent historical stock data using Yahoo Finance.
3. The trained LSTM model processes the previous 60 closing-price observations.
4. The model generates a numerical price prediction.
5. A volatility-based heuristic calculates a confidence label and estimated price band.
6. Groq or Gemini generates a natural-language explanation.
7. Users can ask follow-up questions about the generated prediction.

## ⚠️ Limitations

- Predictions are estimates and are not guaranteed to reflect actual market prices.
- Confidence is a volatility-based heuristic, not a statistically calibrated probability.
- The application requires an internet connection for live stock data and AI API access.
- The current prediction context is stored in memory and is intended for a single-demo-user setup.

## 📌 Disclaimer

This project is developed for educational and demonstration purposes only. It does not provide financial or investment advice. Always conduct independent research before making investment decisions.

## 👨‍💻 Author

Developed as a Machine Learning and Generative AI project.