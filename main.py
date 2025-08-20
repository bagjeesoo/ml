from fastapi import FastAPI, HTTPException
import joblib  # 스케일러 로드를 위해 여전히 필요합니다.
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pydantic import BaseModel

import torch  # 추가
import torch.nn as nn  # 추가
from sklearn.preprocessing import MinMaxScaler  # 스케일러를 위해 추가

# 내부 모듈 임포트
from data_collector import get_historical_crypto_prices, get_recent_news_data
from text_processor import process_news_for_sentiment

# --- LSTM 모델 하이퍼파라미터 설정 (model_trainer.py와 동일해야 합니다) ---
SEQUENCE_LENGTH = 10  # 과거 10일 데이터를 바탕으로 다음 날 예측
HIDDEN_SIZE = 50  # LSTM 계층의 은닉 상태(hidden state) 크기
NUM_LAYERS = 2  # LSTM 계층의 수


# --- PyTorch LSTM 모델 정의 (model_trainer.py에서 가져옴) ---
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class PriceRequest(BaseModel):
    current_coin_price: float


app = FastAPI(
    title="가상화폐 주가 예측 API",
    description="LLM 감성을 활용한 가상화폐 다음 날 주가 예측 API"
)

# 학습된 모델 및 스케일러 로드
model = None
scaler_features = None
scaler_target = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # 1. 모델 아키텍처 인스턴스화
    # input_size는 model_trainer.py에서 2 (close, daily_sentiment_avg)
    model = LSTMPredictor(input_size=2, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=1).to(device)

    # 2. 저장된 state_dict 로드
    model_state_dict = torch.load('crypto_price_prediction_model_lstm.pt', map_location=device)
    model.load_state_dict(model_state_dict)
    model.eval()  # 모델을 평가 모드로 설정
    print("예측 모델 로드 완료.")

    # 3. 스케일러 로드
    scaler_features = joblib.load('scaler_features.joblib')
    scaler_target = joblib.load('scaler_target.joblib')
    print("스케일러 로드 완료.")

except FileNotFoundError as fnfe:
    print(f"\n[오류 발생] 필요한 파일이 없습니다: {fnfe}")
    print("           'python model_trainer.py'를 실행하여 모델과 스케일러를 먼저 학습시키세요.")
except Exception as e:  # FileNotFoundError 외의 모든 다른 예외를 잡습니다.
    print(f"\n[치명적 오류 발생] 모델 또는 스케일러 로딩 중 예상치 못한 오류가 발생했습니다: {e}")
    print("               이 오류는 파일 손상, 버전 불일치, 또는 권한 문제일 수 있습니다.")


@app.get("/")
async def read_root():
    return {"message": "가상화폐 주가 예측 API에 오신 것을 환영합니다! /predict 로 예측을 시도해보세요."}


@app.post("/predict")
async def predict_next_day_price(request_data: PriceRequest):
    """
    주어진 현재 코인 가격과 최신 뉴스 감성을 바탕으로
    다음 날의 가상화폐 종가를 예측합니다.
    """
    current_coin_price = request_data.current_coin_price

    if model is None or scaler_features is None or scaler_target is None:
        raise HTTPException(status_code=500, detail="예측 모델 또는 스케일러가 로드되지 않았습니다. 관리자에게 문의하세요.")

    print(f"\n[Predict Request Received] Current Price: {current_coin_price}")

    # 1. 최신 뉴스 데이터 수집 (오늘 날짜 데이터만)
    latest_news_df = get_recent_news_data(days=1)  # 현재 시점에서 1일치 뉴스

    if latest_news_df.empty:
        print("최신 뉴스 데이터를 찾을 수 없습니다. 중립적인 감성으로 처리합니다.")
        daily_sentiment_avg = 0.0
    else:
        processed_sentiment_df = process_news_for_sentiment(latest_news_df)
        today_date = date.today()
        today_sentiment_df = processed_sentiment_df[processed_sentiment_df['timestamp'].dt.date == today_date]
        if today_sentiment_df.empty:
            print("오늘 날짜에 해당하는 뉴스 감성 데이터를 찾을 수 없습니다. 중립적인 감성으로 처리합니다.")
            daily_sentiment_avg = 0.0
        else:
            daily_sentiment_avg = today_sentiment_df['sentiment_score'].mean()

    print(f"계산된 오늘 날짜 감성 평균: {daily_sentiment_avg:.4f}")

    # --- 2. 예측을 위한 시퀀스 데이터 준비 ---
    # LSTM은 SEQUENCE_LENGTH 만큼의 과거 데이터가 필요합니다.
    # 여기서는 예시를 위해 과거 데이터를 모의로 생성합니다.
    # 실제 환경에서는 데이터베이스에서 과거 'close' 가격과 'daily_sentiment_avg'를 가져와야 합니다.
    mock_historical_data = []
    # SEQUENCE_LENGTH - 1 만큼의 과거 모의 데이터 생성
    for i in range(SEQUENCE_LENGTH - 1, 0, -1):  # SEQUENCE_LENGTH-1 부터 1까지 역순 (가장 최근이 0번째)
        past_price = current_coin_price * (1 + (np.random.rand() - 0.5) * 0.01)  # 현재 가격의 +-0.5% 변동
        past_sentiment = np.random.uniform(-0.1, 0.1)  # 랜덤 감성
        mock_historical_data.append([past_price, past_sentiment])

    # 오늘 데이터 추가
    mock_historical_data.append([current_coin_price, daily_sentiment_avg])

    # NumPy 배열로 변환
    features_np = np.array(mock_historical_data, dtype=np.float32)  # (SEQUENCE_LENGTH, 2)

    # 3. 특성 스케일링
    # features_np는 (SEQUENCE_LENGTH, input_size) 형태.
    # scaler_features는 (샘플 수, 특성 수) 형태의 2D 배열을 기대하므로,
    # 현재는 features_np가 이미 (10, 2) 이므로 바로 전달
    scaled_features = scaler_features.transform(features_np)

    # 4. PyTorch 텐서로 변환 (batch_size, sequence_length, input_size)
    # 현재는 1개의 배치에 대한 시퀀스이므로, 차원 추가
    input_tensor = torch.from_numpy(scaled_features).float().unsqueeze(0).to(device)

    # 5. 다음 날 가격 예측 (torch.no_grad()로 추론 모드 활성화)
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():
        predicted_scaled_price = model(input_tensor).cpu().numpy()

    # 6. 예측값을 원본 스케일로 되돌리기 (Inverse Transform)
    # scaler_target은 1D 배열을 처리할 수 있으므로 predicted_scaled_price도 reshape
    predicted_price_original_scale = scaler_target.inverse_transform(predicted_scaled_price.reshape(-1, 1))[0][0]

    return {
        "status": "success",
        "current_price": current_coin_price,
        "daily_sentiment_score": round(daily_sentiment_avg, 4),
        "predicted_next_day_close_price": float(round(predicted_price_original_scale, 2))
    }
