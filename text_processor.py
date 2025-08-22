import re
import pandas as pd
from datetime import datetime, timedelta

def clean_text(text: str):
    """
    텍스트를 전처리합니다. (예: 특수문자 제거, 공백 정규화)
    """
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text) # 한글, 영어, 숫자, 공백만 남김
    text = re.sub(r'\s+', ' ', text).strip() # 다중 공백 단일 공백으로
    return text

def get_llm_sentiment_score(text: str) -> float:
    """
    주어진 텍스트에 대해 LLM이 감성 점수를 반환하는 것을 모의합니다.
    실제로는 OpenAI, Gemini, Claude 등의 API를 호출하거나
    로컬 LLM (예: Sentence-BERT, KoBERT 등을 fine-tuning한 모델)을 사용합니다.
    """
    cleaned_text = clean_text(text)

    # 이 부분은 실제 LLM API 호출 또는 로컬 모델 추론 로직으로 대체되어야 합니다.
    # 예시를 위해 단순 키워드 기반으로 감성 점수를 할당합니다.
    positive_keywords = ['상승', '급등', '긍정적', '성장', '호재', '개발', '출시'
                        , '혁신', '유입', '강세', '투자', '호황', '금리인하']

    negative_keywords = ['하락', '급락', '부정적', '경고', '규제', '불확실성', '악재'
                        , '약세', '축소', '과열', '금지', '과세', '금리인상', '불황']

    score = 0.0
    text_lower = cleaned_text.lower() # 소문자 변환 후 키워드 매칭

    for keyword in positive_keywords:
        if keyword in text_lower:
            score += 0.2 # 긍정 키워드 발견 시 점수 증가
    for keyword in negative_keywords:
        if keyword in text_lower:
            score -= 0.2 # 부정 키워드 발견 시 점수 감소

    # 점수를 -1에서 1 사이로 정규화 (최소/최대치를 넘지 않도록)
    score = max(-1.0, min(1.0, score))

    return score

def process_news_for_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    뉴스 데이터프레임의 각 뉴스에 대해 감성 점수를 계산합니다.
    """
    # 'title'과 'content'를 합쳐서 감성 분석에 사용할 텍스트 생성
    news_df['full_text'] = news_df['title'] + " " + news_df['content']
    news_df['sentiment_score'] = news_df['full_text'].apply(get_llm_sentiment_score)
    return news_df[['timestamp', 'sentiment_score']]

if __name__ == "__main__":
    # 텍스트 처리 테스트
    sample_text_positive = "비트코인, 기관 투자 유입으로 새로운 최고가 경신"
    sample_text_negative = "미국 규제 당국, 가상화폐 시장에 강력한 경고 메시지"
    sample_text_neutral = "가상화폐 시장, 큰 변동성 없이 보합세 유지"

    print(f"'{sample_text_positive}' 감성 점수: {get_llm_sentiment_score(sample_text_positive)}")
    print(f"'{sample_text_negative}' 감성 점수: {get_llm_sentiment_score(sample_text_negative)}")
    print(f"'{sample_text_neutral}' 감성 점수: {get_llm_sentiment_score(sample_text_neutral)}")

    # 모의 뉴스 데이터 처리
    from data_collector import get_recent_news_data
    mock_news_data = get_recent_news_data(days=2)
    processed_sentiment = process_news_for_sentiment(mock_news_data)
    print("\n처리된 뉴스 감성 점수 (일부):")
    print(processed_sentiment.head())