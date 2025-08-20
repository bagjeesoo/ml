import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib  # scaler 저장을 위해 여전히 필요합니다.
from datetime import datetime, timedelta

from data_collector import get_historical_crypto_prices, get_recent_news_data
from text_processor import process_news_for_sentiment

# --- LSTM 모델 하이퍼파라미터 설정 ---
SEQUENCE_LENGTH = 10  # 과거 10일(SEQUENCE_LENGTH) 데이터를 바탕으로 다음 날(1일) 예측
HIDDEN_SIZE = 50  # LSTM 계층의 은닉 상태(hidden state) 크기
NUM_LAYERS = 2  # LSTM 계층의 수
NUM_EPOCHS = 100  # 학습 반복 횟수
LEARNING_RATE = 0.001  # 학습률


# --- 1. LSTM 시퀀스 데이터 생성 함수 ---
def create_lstm_sequences(features, target, sequence_length):
    """
    주어진 특성과 타겟 데이터로부터 LSTM 학습을 위한 시퀀스를 생성합니다.
    features: (샘플 수, 특성 수) 형태의 NumPy 배열
    target: (샘플 수, 1) 형태의 NumPy 배열 (다음 날 종가)
    sequence_length: 각 시퀀스의 길이
    """
    xs, ys = [], []
    for i in range(len(features) - sequence_length):
        x = features[i:(i + sequence_length)]  # 현재 시퀀스 데이터
        y = target[i + sequence_length]  # 시퀀스 다음 날의 타겟 (이미 shift된 값)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# --- 2. PyTorch LSTM 모델 정의 ---
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # batch_first=True: 입력 텐서의 형태가 (batch_size, sequence_length, input_size)가 되도록 설정
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # LSTM의 마지막 은닉 상태(hidden state)를 받아 최종 출력 값을 생성하는 선형 계층
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # h0, c0는 LSTM의 초기 은닉 상태와 셀 상태입니다. 학습 가능한 파라미터는 아니므로 0으로 초기화
        # shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 계층에 입력 데이터와 초기 상태 전달
        # out: (batch_size, sequence_length, hidden_size) - 시퀀스의 각 타임스텝에서의 출력
        # (hn, cn): 최종 은닉 상태와 셀 상태 튜플
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # 예측에는 시퀀스의 마지막 타임스텝의 출력만을 사용합니다.
        # out[:, -1, :]는 마지막 타임스텝의 모든 샘플에 대한 출력을 의미합니다.
        out = self.fc(out[:, -1, :])
        return out


def train_prediction_model():
    print("모델 학습 시작...")

    # 1. 데이터 수집 (역사적 주가 + 뉴스)
    historical_prices_df = get_historical_crypto_prices(days=200)  # LSTM은 더 많은 데이터가 필요
    all_news_df = get_recent_news_data(days=200)  # LSTM은 더 많은 데이터가 필요

    # 2. 뉴스 감성 분석
    sentiment_df = process_news_for_sentiment(all_news_df)

    # 날짜별 감성 점수 집계 (일별 평균 감성 점수)
    sentiment_df['date'] = sentiment_df['timestamp'].dt.date
    daily_sentiment_avg = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
    daily_sentiment_avg.rename(columns={'sentiment_score': 'daily_sentiment_avg'}, inplace=True)

    # 3. 주가 데이터와 감성 점수 병합 및 특성/타겟 생성
    merged_df = pd.merge(historical_prices_df, daily_sentiment_avg, on='date', how='left')
    merged_df['daily_sentiment_avg'] = merged_df['daily_sentiment_avg'].fillna(0)  # 결측치는 0으로 채움

    # 미래 예측을 위해 타겟 변수(y) 생성: 다음 날의 종가
    # merged_df.iloc[i].close 로 merged_df.iloc[i+1].close 를 예측
    # 즉, 현재 행의 'target_close'는 다음 날의 'close' 값
    merged_df['target_close'] = merged_df['close'].shift(-1)

    # 마지막 시퀀스 생성을 위해 충분한 데이터가 있어야 함. SEQUENCE_LENGTH 만큼 제거
    merged_df.dropna(inplace=True)

    # LSTM 입력으로 사용할 특성 (현재 종가와 일별 감성 평균)
    # NumPy 배열로 변환
    features = merged_df[['close', 'daily_sentiment_avg']].values
    # 타겟 데이터 (다음 날 종가). 스케일링을 위해 2D 배열로 변환
    target = merged_df['target_close'].values.reshape(-1, 1)

    # --- 4. 데이터 스케일링 (MinMaxScaler 사용) ---
    scaler_features = MinMaxScaler(feature_range=(0, 1))  # 특성 스케일러
    scaled_features = scaler_features.fit_transform(features)

    scaler_target = MinMaxScaler(feature_range=(0, 1))  # 타겟 스케일러
    scaled_target = scaler_target.fit_transform(target)

    # --- 5. LSTM 시퀀스 생성 ---
    X_seq, y_seq = create_lstm_sequences(scaled_features, scaled_target, SEQUENCE_LENGTH)

    # --- 6. 학습 및 테스트 세트 분리 (시간 순서 중요) ---
    # 시간 순서를 유지하기 위해 shuffle=False
    train_size = int(len(X_seq) * 0.8)  # 80%를 학습 데이터로 사용
    X_train_tensor = torch.from_numpy(X_seq[:train_size]).float()
    y_train_tensor = torch.from_numpy(y_seq[:train_size]).float()

    X_test_tensor = torch.from_numpy(X_seq[train_size:]).float()
    y_test_tensor = torch.from_numpy(y_seq[train_size:]).float()

    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습에 사용할 디바이스: {device}")

    # --- 7. 모델 인스턴스화 및 학습 준비 ---
    input_size = X_train_tensor.shape[2]  # 특성의 개수 (close, daily_sentiment_avg -> 2개)
    output_size = 1  # 예측할 값의 개수 (다음 날 종가 -> 1개)

    model = LSTMPredictor(input_size, HIDDEN_SIZE, NUM_LAYERS, output_size).to(device)

    criterion = nn.MSELoss()  # 평균 제곱 오차를 손실 함수로 사용
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer 사용

    print("\nLSTM 모델 학습 시작...")
    # --- 8. 모델 학습 루프 ---
    for epoch in range(NUM_EPOCHS):
        model.train()  # 모델을 훈련 모드로 설정
        optimizer.zero_grad()  # 옵티마이저의 기울기 초기화

        # 모델의 순전파 (forward pass)
        outputs = model(X_train_tensor.to(device))
        loss = criterion(outputs, y_train_tensor.to(device))  # 손실 계산

        loss.backward()  # 역전파 (backward pass) - 기울기 계산
        optimizer.step()  # 옵티마이저 스텝 - 가중치 업데이트

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.6f}')  # 손실 출력

    print("LSTM 모델 학습 완료.")

    # --- 9. 모델 평가 ---
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 기울기 계산 비활성화 (메모리 절약, 연산 속도 향상)
        train_preds_scaled = model(X_train_tensor.to(device)).cpu().numpy()
        test_preds_scaled = model(X_test_tensor.to(device)).cpu().numpy()

    # 예측값을 원본 스케일로 되돌리기 (Inverse Transform)
    train_preds = scaler_target.inverse_transform(train_preds_scaled)
    y_train_original = scaler_target.inverse_transform(y_train_tensor.cpu().numpy())
    test_preds = scaler_target.inverse_transform(test_preds_scaled)
    y_test_original = scaler_target.inverse_transform(y_test_tensor.cpu().numpy())

    # RMSE 계산
    rmse_train = np.sqrt(mean_squared_error(y_train_original, train_preds))
    rmse_test = np.sqrt(mean_squared_error(y_test_original, test_preds))
    print(f"\n훈련 세트 RMSE: {rmse_train:.2f}")
    print(f"테스트 세트 RMSE: {rmse_test:.2f}")

    # --- 10. 학습된 모델 및 스케일러 저장 ---
    # PyTorch 모델은 state_dict()를 저장하는 것이 일반적
    model_filename = 'crypto_price_prediction_model_lstm.pt'
    torch.save(model.state_dict(), model_filename)
    print(f"LSTM 모델이 '{model_filename}' (으)로 저장되었습니다.")

    # 스케일러도 나중에 예측 시 필요하므로 저장합니다.
    joblib.dump(scaler_features, 'scaler_features.joblib')
    joblib.dump(scaler_target, 'scaler_target.joblib')
    print("스케일러들이 'scaler_features.joblib', 'scaler_target.joblib' (으)로 저장되었습니다.")

    # 반환할 것이 있다면 (여기서는 파일로 저장했으므로 특별히 반환할 필요는 없습니다.)
    # return model


if __name__ == "__main__":
    train_prediction_model()
    print("\n모델 학습 및 저장 완료.")