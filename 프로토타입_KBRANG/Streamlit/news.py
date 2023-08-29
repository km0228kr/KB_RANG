import streamlit as st
import os
import pandas as pd
import textwrap
import random
from utils.preprocessing import UserLabelProcessor
from models.neural_collaborative_filtering import model_valid

## news_rec - 기사추천
def news_rec(input_csv_path, user_csv_path, output_csv_path, id):
    # Preprocess user information and save to a CSV file
    news = pd.read_csv("./data/news_db.csv") # news 데이터
    processor = UserLabelProcessor(input_csv_path, user_csv_path, output_csv_path)
    processor.melt_and_save()
    
    data_path = "./data/" 

    config = {
        "num_factors": 12,
        "hidden_layers": [256, 256, 256],
        "embedding_dropout": 0.02,
        "dropouts": [0.3, 0.3, 0.3],
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "batch_size": 1024,
        "num_epochs": 300,
        "total_patience": 30,
        "save_path": "params2.data"
        }

    news_id_list = ["금융", "증시", "부동산", "국제경제", "소비자", "경제/정책"]
    user_id = id # 해당 유저의 구분
    user_id_list = [user_id] * len(news_id_list)
    pred_results = model_valid(user_id_list, news_id_list, data_path, config)
    
    # 데이터프레임 만들기
    result_df = pd.DataFrame({
        "userId": user_id_list,
        "label": news_id_list,
        "pred_ratings": [float(r.detach().numpy()) for r in pred_results],
    })

    low_rec_label = result_df.sort_values(by=["pred_ratings"], ascending=[False]).iloc[-1, 1] # 취약한 카테고리
    high_rec_label = result_df.sort_values(by=["pred_ratings"], ascending=[False]).iloc[0, 1] # 관심있는 카테고리
    st.session_state.low_rec_label = low_rec_label
    st.session_state.high_rec_label = high_rec_label
    rec_labels = [low_rec_label] + [high_rec_label]
    st.session_state.rec_labels = rec_labels
    
    # 뉴스 추천
    # news의 라벨이 rec_label인 것들만 특정 후 인덱스 초기화
    low_rec_news_zip = news[news["label"]==low_rec_label].reset_index(drop=True) 
    high_rec_news_zip = news[news["label"]==high_rec_label].reset_index(drop=True)
    
    # 무작위 6개의 기사 추출 
    low_news_idx = random.sample(low_rec_news_zip.index.tolist(), 3)
    high_news_idx = random.sample(high_rec_news_zip.index.tolist(), 3)

    
    ## 추천된 뉴스 title과 content 구분
    low_news_titles = low_rec_news_zip.loc[low_news_idx, "title"].to_list() # 뉴스 제목 리스트 반환 news_idx
    low_news_contents = low_rec_news_zip.loc[low_news_idx, "content"].to_list() # 뉴스 본문 리스트 반환 news_idx
    
    high_news_titles = high_rec_news_zip.loc[high_news_idx, "title"].to_list() # 뉴스 제목 리스트 반환 news_idx
    high_news_contents = high_rec_news_zip.loc[high_news_idx, "content"].to_list() # 뉴스 본문 리스트 반환 news_idx
    
    news_titles = low_news_titles + high_news_titles
    news_contents = low_news_contents + high_news_contents
    
    
    return low_rec_label, high_rec_label, news_titles, news_contents # 모두 low + high

def rec_list(news_title, news_content):
    
    ## 기사 출력
    st.write(f"### {news_title}")
    
    ## 반복문으로 출력
    contents = news_content.split('다.') ## '-다.' 로 끊기
    for idx, segment in enumerate(contents): # '-다.' 단위로 끊어온 것을 차례로 내보낸다.
        if idx < len(contents) -1: # 마지막 부분에는 붙이지 않기 위해
            txt = textwrap.fill(segment+"다.", width=28)
            st.text(txt)
        else:
            txt = textwrap.fill(segment, width=28)
            st.text(txt)