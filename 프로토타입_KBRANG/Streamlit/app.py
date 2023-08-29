# 시연 영상용
## 사이드바에 기사 정렬
### 일렬로 정렬 완료 - 기사추천은 params의 수정이 필요
import streamlit as st
import os
import time
from user import user_pick, update_user_db, update_click_rate, update_select_db, see_graph
from quiz import qa, check
from news import news_rec, rec_list

# define path
data_path = os.path.join(os.getcwd(), "data")
user_path = os.path.join(data_path, "user_db.csv")

## show_predict_page
def show_predict_page():
    ## sidebar 등장시 자동 등장 및 크기 고정
    st.set_page_config(initial_sidebar_state='expanded')
    
    if not hasattr(st.session_state, 'start'):
        # 시작 이미지를 화면에 표시
        st.session_state.start = True # 시작 이미지는 초기 화면에만 등장
        
        # 중간 쯤에 시작 로고가 보이게 하기위해 구역 지정
        empty1, col1, empty2 = st.columns([0.3, 1.0, 0.3])
        empty1, col2, empty2 = st.columns([0.3, 1.0, 0.3])
        empty1, col3, empty2 = st.columns([0.3, 1.0, 0.3])
        with empty1:
            pass
        with col1:
            pass        
        with col2:
            start_image = st.empty()
            start_image.image('./data/img/start_logo.png')
            time.sleep(2) # 잠시 대기
            start_image.empty()
        with col3:
            pass
        with empty2:
            pass
    st.empty() # 화면 비우기
    
    # 스타일 시트 파일을 열고 읽어서 스타일 적용
    with open('style.css', "r", encoding="utf-8",) as f: # 상단의 색 고정
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    st.image("./data/img/header_banner.png") # 항상 보이는 로고
    
    if not hasattr(st.session_state, 'user_id'): # 상태를 지정하여 선택은 한 번만 되도록 설정
        user_pick(user_path) # 일단 user_id만 사용
        
    if not hasattr(st.session_state, 'quiz_id'):
        qa() # 퀴즈 출제 - 이후로는 퀴즈가 출제되지 않고 저장된 값만 보여준다.
        
    if not hasattr(st.session_state, 'rec_labels'):
        st.title("Quiz")
        st.write(st.session_state.qa_txt) # 저장된 문제 출력
        user_ans = st.radio("정답", ("O", "X")) # 정답 표시
        if user_ans == "O":
            user_ans = 0
        else:
            user_ans = 1
    
        qa_submit = st.button("submit", key="qa_button") # 버튼 생성
        
        if qa_submit:
            st.session_state.user_ans = user_ans 
            
            ## 정답 여부 판별
            check(st.session_state.qa_ans, st.session_state.user_ans) 

            # 스타일을 적용한 CSS 파일 불러오기
            with open('style.css', encoding="utf-8") as f: # 가로로 정답을  나타내는 사진과 해설을 출력
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            
                # 가로로 나열되는 컨테이너 생성
                st.markdown('<div class="css-keje6w">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])  # 화면을 두 개의 열로 분할
                with col1:
                    st.write("### 정답: ")
                    st.image(st.session_state.img, width=50)  # 이미지 표시
                    
                with col2:
                    st.write("#### 해설")
                    st.write(st.session_state.qa_info)
                # 컨테이너 종료
                st.markdown('</div>', unsafe_allow_html=True)   
            
        
    if hasattr(st.session_state, 'user_ans') and not hasattr(st.session_state, 'rec_labels'):
        ## user 업데이트 - 정답률을 반영하기에 1차적으로 업데이트
        user_db = update_user_db(st.session_state.qa_label, st.session_state.ox, st.session_state.idx, user_path, data_path)
        
        see_graph(user_db) # 그래프 출력
        with st.spinner('뉴스를 추천 중입니다...'): # 뉴스를 추천해주는 동안 떠 있을 문구
        
            low_rec_label, high_rec_label, news_titles, news_contents = news_rec(os.path.join(data_path, "split_db.csv"),
                                                                os.path.join(data_path, "split_db.csv"),
                                                                os.path.join(data_path, "rating.csv"),
                                                                st.session_state.user_id)
            st.session_state.news_titles = news_titles
            st.session_state.news_contents = news_contents
            
            # 뉴스가 추천되었으면 해당 라벨들의 추천 횟수 각각 3회씩 증가
            update_select_db(low_rec_label, high_rec_label, st.session_state.idx, user_path, data_path)
            
            # 재정의 되어  초기화 되는 것을 방지하기 위하여 미리 생성 - 다른 뉴스기사를 클릭하여도 재생성되지 않는다.
            st.session_state.low_click = []
            st.session_state.high_click = []
        
    if hasattr(st.session_state, 'rec_labels') and hasattr(st.session_state, 'user_ans'):
        st.markdown( # 사이드 바의 크기 고정
                    """
                <style>
                [data-testid="stSidebar"][aria-expanded="true"]{
                    min-width: 200px;
                    max-width: 200px;
                }
                """,
                    unsafe_allow_html=True,
                )
        
        ## title 6개 출력
        st.sidebar.title(f"📰추천드리는 카테고리: {st.session_state.rec_labels[0]}") # 사용자가 약한 카테고리
        for news_idx, news_title in enumerate(st.session_state.news_titles[:3]):
            unique_key = f"sidebar_button_{news_idx}"           # 고유값 지정
            if st.sidebar.button(news_title, key=unique_key): # 고유값에 따른 버튼
                
                # 해당 기사를 클릭하면 중복되지 않게 해당 카테고리의 클릭 수 증가
                update_click_rate(st.session_state.idx, st.session_state.rec_labels[0], 
                                  user_path, data_path, news_idx, st.session_state.low_click)
                st.session_state.news_title = news_title
                st.session_state.news_content = st.session_state.news_contents[news_idx]
                rec_list(st.session_state.news_title, st.session_state.news_content)
                 
        st.sidebar.title(f"📰추천 카테고리: {st.session_state.rec_labels[1]}")   # 사용자가 강한 카테고리
        for news_idx, news_title in enumerate(st.session_state.news_titles[3:]):
            unique_key = f"sidebar_button_{news_idx+3}" 
            if st.sidebar.button(news_title, key=unique_key):
                
                # 해당 기사를 클릭한 경우 중복되지 않게 해당 카테고리의 클릭 수 증가
                update_click_rate(st.session_state.idx, st.session_state.rec_labels[1],
                                  user_path, data_path, news_idx+3, st.session_state.high_click)
                st.session_state.news_title = news_title
                st.session_state.news_content = st.session_state.news_contents[news_idx+3]
                rec_list(st.session_state.news_title, st.session_state.news_content)
                