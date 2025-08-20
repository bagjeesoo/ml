import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_historical_crypto_prices(coin: str = "BTC", days: int = 30):
    """
    주어진 코인에 대한 과거 가격 데이터를 모의로 생성합니다.
    실제 프로젝트에서는 ccxt 라이브러리나 거래소 API를 사용하여 수집합니다.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    # 가상의 종가(close price) 데이터 생성 (임의의 변동성을 가집니다)
    base_price = 40000.0 if coin == "BTC" else 2500.0 # 초기 가격 설정
    prices = [base_price + np.random.randn() * 1000 for _ in range(days)]
    prices = np.array(prices).cumsum() + base_price # 누적합으로 시계열처럼 보이게

    # 양수 값만 가지도록 조정
    prices = np.maximum(prices, 1000.0)

    df = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    df['date'] = df['date'].dt.date # 날짜만 유지
    return df

def get_recent_news_data(days: int = 1):
    """
    최근 뉴스 데이터를 모의로 생성합니다.
    실제 프로젝트에서는 웹 스크래핑이나 뉴스 API를 사용하여 수집합니다.
    """
    # 가상의 뉴스 데이터 목록
    mock_news = [
        {"timestamp": datetime.now() - timedelta(hours=1), "title": "비트코인, 기관 투자자 유입으로 상승세 지속", "content": "대형 자산운용사들의 비트코인 ETF 투자 확대 소식에 가격이 급등하고 있습니다."},
        {"timestamp": datetime.now() - timedelta(hours=2), "title": "이더리움 개발자, 확장성 개선 로드맵 발표", "content": "이더리움 재단이 새로운 샤딩 기술을 공개하며 네트워크 효율성 증대를 예고했습니다."},
        {"timestamp": datetime.now() - timedelta(hours=3), "title": "알트코인 시장, 과열 경고음…주의 필요", "content": "일부 알트코인의 급격한 가격 상승에 전문가들은 투자에 신중해야 한다고 조언했습니다."},
        {"timestamp": datetime.now() - timedelta(hours=4), "title": "미국 중앙은행, 가상자산 규제 강화 시사", "content": "새로운 디지털 자산 규제 법안 발의 가능성이 제기되며 시장의 불확실성이 커지고 있습니다."},
        {"timestamp": datetime.now() - timedelta(hours=5), "title": "탈중앙 금융(DeFi) 생태계, TVL(총 예치 자산) 사상 최고치 경신", "content": "디파이 플랫폼들의 성장세가 가속화되고 있으며, 사용자 유입이 활발합니다."},
    ]

    # 과거 날짜에 대한 뉴스 추가 (학습 데이터용)
    for i in range(1, days + 1):
        past_date = datetime.now() - timedelta(days=i)
        mock_news.append({
            "timestamp": past_date - timedelta(hours=np.random.randint(1,24)),
            "title": f"과거 뉴스 제목 {i}: 비트코인 관련 { '긍정적' if np.random.rand() > 0.5 else '부정적'} 이슈",
            "content": f"이것은 {past_date.strftime('%Y-%m-%d')}의 가상 뉴스 내용입니다. 시장 분위기는 대체로 { '낙관적' if np.random.rand() > 0.5 else '비관적'}입니다."
        })


    # 요청된 일수에 해당하는 데이터만 필터링
    cutoff_time = datetime.now() - timedelta(days=days)
    recent_news = [news for news in mock_news if news['timestamp'] >= cutoff_time]

    return pd.DataFrame(recent_news)

if __name__ == "__main__":
    # 데이터 수집 테스트
    crypto_data = get_historical_crypto_prices(days=60)
    print("역사적 주가 데이터 (일부):")
    print(crypto_data.tail())

    news_data = get_recent_news_data(days=5)
    print("\n최근 뉴스 데이터 (일부):")
    print(news_data.head())