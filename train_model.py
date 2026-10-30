import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os

# ====== Parameters ======
STOCK = "AAPL"
SEQ_LEN = 60
EPOCHS = 20
BATCH_SIZE = 32
LR = 0.001

# ====== Get Current Folder ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ====== 1. Fetch Data ======
data = yf.download(STOCK, start="2015-01-01", end="2025-01-01")
close_prices = data['Close'].values.reshape(-1, 1)

# ====== 2. Normalize ======
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(close_prices)

# ====== 3. Create Sequences ======
X, y = [], []
for i in range(SEQ_LEN, len(scaled)):
    X.append(scaled[i - SEQ_LEN:i, 0])
    y.append(scaled[i, 0])
X, y = np.array(X), np.array(y)
X = np.expand_dims(X, axis=2)

# ====== 4. PyTorch Dataset ======
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ====== 5. LSTM Model ======
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ====== 6. Training ======
print("Training started...\n")
for epoch in range(EPOCHS):
    total_loss = 0
    for X_batch, y_batch in loader:
        output = model(X_batch).squeeze()
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss / len(loader):.6f}")

# ====== 7. Save Model ======
model_path = os.path.join(BASE_DIR, "model.pth")
scaler_path = os.path.join(BASE_DIR, "scaler.save")

torch.save(model.state_dict(), model_path)
joblib.dump(scaler, scaler_path)

print(f"\n✅ Model trained and saved successfully!")
print(f"📁 Model Path: {model_path}")
print(f"📁 Scaler Path: {scaler_path}")
