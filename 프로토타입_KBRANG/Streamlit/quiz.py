import os
import pandas as pd
import streamlit as st
import random
import textwrap

## pick question
def qa():
    quiz = pd.read_csv("./data/e_ox.csv") # 문제 데이터 # 문답 데이터 호출
    
    ## 문제 선택
    quiz_id = random.randint(0, quiz.shape[0]-1) # 0부터 해당 열의 갯수만큼 중 랜덤
        
    qa_quiz = quiz.loc[quiz_id, "question"] # 질문
    qa_ans = quiz.loc[quiz_id, "answer"] # 정답
    qa_label = quiz.loc[quiz_id, "label"] # 카테고리
    qa_info = quiz.loc[quiz_id, "info"] # 간단한 해설
        
    ## st.session_state 에 저장
    st.session_state.quiz_id = quiz_id # id 고정
    st.session_state.qa_quiz = qa_quiz
    st.session_state.qa_ans = qa_ans
    st.session_state.qa_label = qa_label
    st.session_state.qa_info = qa_info

    qa_txt = textwrap.fill(st.session_state.qa_quiz, width=30) # 최대 30자로 길이 제한
    st.session_state.qa_txt = qa_txt

## question_check
def check(qa_ans, user_ans):
    image_O = "./data/img/red_O_128.png"
    image_X = "./data/img/red_X_128.png"
    if qa_ans == 0: # user의 정답과 실제 정답 비교 후 문구생성
        ox = 0
        img = image_O
    else:
        ox = 1
        img = image_X
    
    if qa_ans == user_ans: # user의 정답과 실제 정답 비교 후 문구생성
        ox = 0
        st.success("정답 입니다.")
    else:
        ox = 1
        st.warning("오답입니다.")
    
    st.session_state.img = img
    st.session_state.ox = ox