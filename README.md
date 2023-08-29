# 제 5회 Future Finance A.I. Challenge

- http://kbdatory.com/notice/view
- 주관 기관 KB 국민은행
- 프로젝트 기간 : 23.07.21 ~ 23.08.20
- 목적 : 금융업 전반에서 인공지능 기술을 활용한 문제개선 및 가치 창출
- **주제 : 생성 AI 모델을 활용한 개인맞춤 경제 뉴스 기사 추천 서비스**
  - 금융 어플리케이션의 MAU (Monthly Acitve User) 및 DAU (Daily Active User)를 증가시키기 위한 컨텐츠 추가 및 하루 한번 접속을 위한 일상 그 자체를 녹여낼 수 있는 컨텐츠 필요
  - **생성 모델을 활용한 경제 문제 및 정답 자동 생성**
  - 생성된 문제를 사용자에게 제공하여 문제의 카테고리별 정답률과 클릭률 및 개인정보(성별, 연령대, 거주지 등)를 통한 개인 맟춤 **경제 뉴스 추천 시스템**
 
- 프로토 타입 링크 : https://lvbw36rfmca4idskyq6jum.streamlit.app/

### 프로토타입 실행 방법

> cd KBRANG_prototype > cd Streamlit > streamlit run main.py
---
# 팀 차동민

|이름|닉네임|프로필|역할|
|:--|:---|:-----|:---:|
|차형석|hsmaro|https://github.com/hsmaro|Streamlit|
|김동하|Eastha0526|https://github.com/Eastha0526|Rec Sys|
|권경민|km0228kr|https://github.com/km0228kr|NLP|

---
# 기술 흐름도
![image](https://github.com/Eastha0526/KB_RANG/assets/110336043/eaee3d04-86cb-4515-9f67-d9c3dc16afcc)


---
# 프로젝트 구조

```
├── KBRANG_prototype/
│   ├── Generation Datasets/ # for dataset
│   │   ├── Data Crawling/
│   │   │   └── news_db_crawling.ipynb  # for news crawling
│   │   ├──user_data.ipynb # for generation imitaion user dataset
│   │   └── aihub_ox_data.ipynb # for preprocessing ox data
│   ├── Models
│   │   ├── Generation model/ # for generation
│   │   │   └── GPT_API_GenQ.ipynb # few-shot learning with OPEN API for Generation Dataset
│   │	│   └── kb_albert.ipynb # generation answer with kb-albert
│   │   │   └── kobert.ipynb # generation answer with kobert
│   │	│   └── KoElectra.ipynb # generation answer with KoElectra
│   ├─ Recommendation System/
│   │   └── models/
│   │   │   └── neural_collaborative_filtering.py # collaboraive filtering with NCF model
│   │   └── utils/
│   │       └── preprocessing.py # preprocessing for user dataset
│   └── Streamlit/  # for prototype
│            └── data/
│            └── app.py  # total app flow
│            └── main.py # to run folder
│            └── news.py # for rec sys
│            └── quiz.py # for quiz
│            └── user.py # for user and update user db
│            └── style.css # for decorating
│ 
├── requrirements.txt
└── README.md
```
---
## 프로젝트 목적

- 고객 기반 확대
- Trendy, Informative, Practical 을 목적으로한 콘텐츠 강화, 하루 한번 접속, 일상 그자체를 녹여내기 위함
- MAU 및 DAU를 확대시켜 인터넷 전문 은행들과의 경쟁
- 리텐션 효과를 통한 고객 충성도 증가

## 서비스 개요
- 경제 상식 문제 제공
  - 사용자의 경우 생성 모델로 만들어진 일반 경제 상식 문제를 풀고 포인트를 얻는다. [하루 한번, 일상 그 자체]
- 개인 맞춤 뉴스 추천 시스템
  - 사용자는 사용자의 개인 정보와 함께 퀴즈 정답률 및 클릭률에 기반한 개인 맞춤 뉴스를 추천 받는다. [콘텐츠 강화, 개인 맞춤 추천]
 
## 데이터셋 구성

- 모의 데이터셋
  - 연령대 : 국민은행 어플 사용 비율 기반으로 한 생성
  - 직업 / 거주지 : 통계청 전수조사 기준 생성
  - 뉴스 : 경제 카테고리 기준으로 추천 레이블 생성
  - 정답률 및 클릭률 : 랜덤하게 정규분포 생성
 
- 뉴스 데이터 크롤링
  - 뉴스 추천의 경우 시의성이 가장 중요
  - 2시간 마다 새로운 뉴스를 수집
    - Bs4, Selenium 활용
   
- 경제 문제 데이터
  - 경제 문제 생성을 위한 Few-shot learning 사용
    - few-shot을 활용하기 위한 문제 출처 : 기획 재정부 경제 배움e 문제, 어린이 백과문제
    - 경제 문제의 경우 매일매일 흥미 위주의 문제 생성을 위하여 일반적인 경제 상식 문제 활용

## 금융 문제 생성

- Open AI의 GPT API를 활용한 문제 생성
- Few-shot learning을 활용하여 api 기반 문제 생성

```python

messages = []
content = prompt
messages.append({"role" : "user", "content" : content})

completion = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages=messages
)

chat_responese = completion.choices[0].message.content
print(f"{chat_responese}")
messages.append({"role" : "assistant", "content" : chat_responese})

while True:
    # 조건을 걸어 원하는 만큼 문제를 계속해서 생성
    content = input("User: ")
    messages.append({"role" : "user", "content" : content})
    # 유저의 입력

    completion = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages=messages
    )

    chat_responese = completion.choices[0].message.content
    print(f"{chat_responese}")
    messages.append({"role" : "assistant", "content" : chat_responese})

    cnt = int(input("계속 문제를 생성할까요? 0: 생성 1: 종료"))

    if cnt == 1:
        break
```

## 문제 정답 생성

- KB-ALBERT 사용한 파인튜닝
- https://github.com/teddylee777/KB-ALBERT-KO
- KB-ALBERT의 경우 경제/금융 도메인에 특화되어 타 PLM모델에 비하여 높은 성능을 보임
- https://huggingface.co/
- hugging face의 transformers 라이브러리 활용

```python
model = AlbertForSequenceClassification.from_pretrained(kb_albert_model_path, num_labels=2)

# Evaluation
model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    y_true = []  # 실제 정답값 저장
    y_pred = []  # 예측값 저장
    for batch in test_dataloader:
        inputs = {k: v.squeeze(1).to('cuda') for k, v in batch['inputs'].items()} # 입력 데이터
        labels = torch.stack([torch.tensor(int(a)).to('cuda') for a in batch['answer']]) # 정답값
        outputs = model(**inputs) # 예측값
        _, predicted = torch.max(outputs.logits, 1) # 예측값 조정
        total += labels.size(0)
        correct += (predicted == labels).sum().item() # 정답률
        y_true.extend(labels.cpu().numpy()) # 실제값 저장
        y_pred.extend(predicted.cpu().numpy()) # 예측값 저장


# y_true와 y_pred를 사용하여 f1 스코어 계산
f1 = f1_score(y_true, y_pred, average='weighted')

print('Accuracy: %d %%' % (100 * correct / total)) #
print('F1 Score:', f1)
```

- 학습의 경우 파인튜닝이기 때문에 비교적 낮은 학습률을 사용
- Adam 옵티마이저와 함께 1e-4의 학습률을 사용한 학습
- 조기종료를 활용한 과적합 방지

## 추천 시스템

- https://arxiv.org/abs/1708.05031
- Neural Collaborative Filtering을 활용한 딥러닝 기반 추천 시스템 모델 제작
- 논문에서는 0과 1의 레이블을 사용하였지만, 현 상황에 맞게 예측 정답값을 예측하는 추천시스템 제작
- User Vector(성별, 나이, 거주지)와 Item Vector(뉴스 레이블), Action Vector(클릭률, 정답률)를 임베딩한 개인화 맞춤 추천 시스템
- 개인 정보와 함께 퀴즈를 통해 맞춘 정답률과 클릭률을 개인의 금융 관심도로 지정하여 개인화 맞춤 뉴스를 추천한다.

```python
class FeedForwardEmbedNN(nn.Module):

    def __init__(self, n_news, hidden, dropouts, n_factors, embedding_dropout):
        super().__init__()
        '''
        neural collaborative filtering 논문 모델 사용
        NCF의 일반적인 딥러닝 모델로 추천 스코어 예측
        유저, 나이, 직업, 거주지의 경우 임베딩 -> 유저 벡터
        뉴스 -> 아이템 벡터
        rating, click_probs -> action 벡터로 지정
        클릭율과, 정답률의 경우 임베딩하지 않고 사용
        '''
        self.news_emb = nn.Embedding(n_news, n_factors)
        self.gender_emb = nn.Embedding(num_genders, n_factors)  
        self.age_emb = nn.Embedding(num_ages, n_factors)  
        self.occupation_emb = nn.Embedding(num_occupations, n_factors) 
        self.address_emb = nn.Embedding(num_addresses, n_factors)  
        
        self.drop = nn.Dropout(embedding_dropout)
        self.hidden_layers = nn.Sequential(*list(self.generate_layers(n_factors*5 + 2, hidden, dropouts))) 
        self.fc = nn.Linear(hidden[-1], 1)

    def generate_layers(self, n_factors, hidden, dropouts):
        assert len(dropouts) == len(hidden)

        idx = 0
        while idx < len(hidden):
            if idx == 0:
                yield nn.Linear(n_factors, hidden[idx])
            else:
                yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(dropouts[idx])

            idx += 1

    def forward(self, news, gender, age, occupation, address, acc_avg, click_probs, min_rating=0., max_rating=1.):
        news_embeds = self.news_emb(news)
        gender_embeds = self.gender_emb(gender)
        age_embeds = self.age_emb(age)
        occupation_embeds = self.occupation_emb(occupation)
        address_embeds = self.address_emb(address)

        acc_avg = torch.tensor(acc_avg)
        acc_avg = acc_avg.unsqueeze(1)

        click_probs = torch.tensor(click_probs)
        click_probs = click_probs.unsqueeze(1)


        concat_features = torch.cat([news_embeds, gender_embeds, age_embeds, occupation_embeds, address_embeds, acc_avg, click_probs], dim=1)
        x = F.relu(self.hidden_layers(concat_features))
        out = torch.sigmoid(self.fc(x))
        out = (out * (max_rating - min_rating)) + min_rating

        return out
```
## 사용자 스터디를 통한 결과 평가

- 실제 프로토타입을 기반으로 한 사용자 스터디를 통한 추천 결과 도출
- "바로 뉴스를 추천해주기 때문에 포털 사이트에서 직접 찾아보는 것 보다 훨씬 편리하다"
- "내 관심 분야 & 관심을 가지면 좋을 분야를 함께 추천해주기에 더 다양한 뉴스를 볼 수 있는 것 같다"
- "매일 일상적으로 즐길 수 잇는 서비스 같다"
- "평소에 일주일에 한 번 정도 들어가는데 해당 서비스가 있으면 맹리 접속할 것 같다"

- 추천 시스템의 경우 실제 모델의 추천 결과보다 사용자 스터디 및 A/B 테스트를 통한 결과를 통해 보완하며 여러 지표를 생성해야 한다.

# 기대효과 및 결론

#### 일상으로 들어온 효과

- 단순히 "금융 업무"가 아닌 사용자의 일상에 녹아드는 서비스는 고객과 장기적인 관계를 구축할 수 있으며 신뢰를 높이는 효과를 가짐

#### 고객 충성도 확보

- 높은 리텐션은 기업의 지속적인 성장을 위한 핵심 요소로 장기적인 고나점에서도 고객 가치를 극대화

#### 개인 맞춤 금융 상품 추천

- 누적된 데이터를 활용하여 제공되는 맞춤 추천 서비스는 고객의 효율적인 금융 선택을 돕고, 기업과 고객 사이의 신뢰도를 높임

#### 종합금융플랫폼으로의 도약

- 종합 금융 플랫폼 전략을 통해 슈퍼앱이 제공하는 다양한 서비스와 함께 통합된 금융 서비스를 제공
- 빅테크와의 경쟁에서 더욱 탄탄한 포지션을 확보
- 고객의 다양한 금융 관련 니즈에 신속하고 효율적으로 응답함으로써, 시장에서의 선도적 위치를 더욱 강화


### 참고 문헌 및 레퍼런스

- He, Xiangnan, Liao, Lizi, Zhang, Hanwang, Nie, Liqiang, Hu, Xia and Chua, Tat-Seng. "Neural Collaborative Filtering.." Paper presented at the meeting of the WWW, 2017.
- 남빛하늘, “이창권號 국민카드, ‘KB페이’ 1000만 고객 확보 비결”, 인사이트코리아, 2023.07.14., http://www.insightkorea.co.kr/news/articleView.html?idxno=114835
- 박기록, "KB국민은행 , 명실상부한 '1위 종합금융플랫폼' 회사가 될 수 있을까", 디지털데일리, 2023.07.28., https://m.ddaily.co.kr/page/view/2023072716575852088
- 에너지경제신문, “신한금융, 비금융 앱 많이 찾는다…MAU 400만명 눈앞”, 에너지경제, 2023.05.01., https://m.ekn.kr/view.php?key=20230428010007595
- 유한일, “시중·인터넷은행 ‘뱅킹앱’ 경쟁···디지털 타고 이용률 쭉쭉”, 뉴스투데이, 2023.05.02., https://www.news2day.co.kr/article/20230428500205
- 정단비, “금융 앱 MAU 2000만 시대···상반기 톱은 '신한'”, 뉴스웨이, 2023.08.10., https://www.newsway.co.kr/news/view?ud=2023081016013505522
- 홍하나, “인터넷은행은 왜 MAU를 강조할까?”, 바이라인네트워크, 2022.08.07., https://byline.network/2022/08/05-64/
- 황양택, “은행부터 보험사까지 '디지털 전환' 몰두”, 뉴스토마토, 2023.07.19., https://www.newstomato.com/ReadNews.aspx?no=1194420
- WISEAPP, “2022년 국내 은행 앱 순위는?”, 2022.08.24., https://www.wiseapp.co.kr/insight/detail/254
