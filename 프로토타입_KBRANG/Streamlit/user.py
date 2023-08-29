import os
import pandas as pd
import streamlit as st
import random
import plotly.express as px

# pick_user
def user_pick(user_path):
    user  = pd.read_csv(user_path) # user의 기본정보
    idx = random.randint(0, user.shape[0]-1)
    id = f"user_{idx+1}" # user_id 생성
    selected_user = user[user["user_id"]==id] # 해당 user의 행만 추출
    st.session_state.selected_user = selected_user 
    
    # 해당 user의 정보 추출
    user_gender = selected_user.loc[idx, "gender"] 
    user_age  = selected_user.loc[idx, "age"]
    user_occupation = selected_user.loc[idx, "occupation"]
    user_address = selected_user.loc[idx, "address"]

    # st.session_state 페이지가 종료되어도 값이 남아있다.
    st.session_state.idx = idx
    st.session_state.user_id = id
    st.session_state.user_gender = user_gender
    st.session_state.user_age = user_age
    st.session_state.user_occupation = user_occupation
    st.session_state.user_address = user_address
    
# user_update
def update_user_db(label, ox, idx, user_path, data_path):
    user_db  = pd.read_csv(user_path) # user의 기본정보
    
    user_db.loc[idx, "total"] += 1 # 전체 출제 수 업데이트
    user_db.loc[idx, f"{label}_tot"] += 1 # 전체 해당 라벨 문제 출제 수 업데이트
    if ox == 0: # 정답일 경우
        user_db.loc[idx, f"{label}_ans"] += 1 # 해당 라벨의 정답횟수 업데이트
        
    # 추가 업데이트 필요 - 클릭율과 정답률의 평균
    user_db.loc[idx, label] = user_db.loc[idx, f"{label}_ans"] / user_db.loc[idx, f"{label}_tot"] # 정답률 업데이트
    
    # 정답률 펑균 업데이트 및 클릭율 업데이트  or "금융":"경제일반"
    user_db.loc[idx, "acc_avg"] = user_db.loc[idx, "금융":"경제/정책"].sum() / 6
        
    selected_user = user_db[user_db["user_id"]==st.session_state.user_id] # 해당 user의 행만 추출
    st.session_state.selected_user = selected_user
    
    split_user_db = user_db.iloc[:, 0:18]
    
    user_db.to_csv(user_path, encoding="utf-8", index=False) ## 나중에 바꾸기
    split_user_db.to_csv(os.path.join(data_path, "split_db.csv"), encoding="utf-8", index=False)
    return user_db

# 카테고리별로 기사가 추천되었을 때 사용
def update_select_db(low_rec_label, high_rec_label, idx, user_path, data_path):
    user_db = pd.read_csv(user_path)
    user_db.loc[idx, f"rec_{low_rec_label}_news"] += 3 # 해당 카테고리에 각각 3개씩 추천 횟수 추가
    user_db.loc[idx, f"rec_{high_rec_label}_news"] += 3
    
    ## 재조정
    user_db.loc[idx, f"{low_rec_label}_click_probs"] = user_db.loc[idx, f"{low_rec_label}_click"] / user_db.loc[idx, f"rec_{low_rec_label}_news"] 
    user_db.loc[idx, f"{high_rec_label}_click_probs"] = user_db.loc[idx, f"{high_rec_label}_click"] / user_db.loc[idx, f"rec_{high_rec_label}_news"]

    # 분할
    split_user_db = user_db.iloc[:, 0:18]

    # 저장
    user_db.to_csv(user_path, encoding="utf-8", index=False)
    split_user_db.to_csv(os.path.join(data_path, "split_db.csv"), encoding="utf-8", index=False)

## 기사를 눌럿을 때 사용
def update_click_rate(idx, label, user_path, data_path, news_idx, click_list):
    user_db = pd.read_csv(user_path)
    if news_idx not in click_list and label == st.session_state.rec_labels[0]:
        st.session_state.low_click.append(news_idx)
        user_db.loc[idx, f"{label}_click"] += 1
        user_db.loc[idx, f"{label}_click_probs"] = user_db.loc[idx, f"{label}_click"] / user_db.loc[idx, f"rec_{label}_news"]
        
    elif news_idx not in click_list and label == st.session_state.rec_labels[1]:
        st.session_state.high_click.append(news_idx)
        user_db.loc[idx, f"{label}_click"] += 1
        user_db.loc[idx, f"{label}_click_probs"] = user_db.loc[idx, f"{label}_click"] / user_db.loc[idx, f"rec_{label}_news"]
    else:
        pass
    
    # 분할
    split_user_db = user_db.iloc[:, 0:18]

    # 저장
    user_db.to_csv(user_path, encoding="utf-8", index=False)
    split_user_db.to_csv(os.path.join(data_path, "split_db.csv"), encoding="utf-8", index=False)


def see_graph(user_db):
     u_df = user_db.iloc[st.session_state.idx:st.session_state.idx+1, 5:11].T.reset_index() 
     # 5:11 -> 금융, 증시 ,부동산, 국제경제, 소비자, 경제/정책
     u_df.columns=["label", "정답률"]
     
     fig = px.bar(u_df, x="label", y="정답률", title=f"{st.session_state.user_id}의 정답률",
                  hover_name = "label",
                  hover_data={"정답률": ":.2f",
                              "label":False})
     fig.update_xaxes(title='') # x축 이름 제거
     fig.update_yaxes(title='') # y축 이름 제거
     fig.update_yaxes(showgrid=False) # y축 그리드(눈금) 제거
     fig.update_layout(font_color='black', # 글씨 색 바꾸기
                       xaxis=dict(tickfont=dict(color='black')), # x 축 눈금 색상 변경  
                       yaxis=dict(tickfont=dict(color='black')),
                       width=350) # y 축 눈금 색상 변경
     fig.update_traces(marker_color='#60544c') # bar 색상 바꾸기
     st.plotly_chart(fig) # 차트랑 문제가 같이 뜨게 설정