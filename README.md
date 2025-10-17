# DATA 주소
https://dacon.io/competitions/official/236562/codeshare



# 주문
1. 변수선택법 중 하나인 연속형 vs 연속형 일떄는 피어슨 상관계수를 쓴다. label이 명목형일때는 연관성을 어떻게 측정하는지 고찰
스피어만 상관계수, 피어슨 상관계수가 뭔지 알아보자
2. kmeans를 통한 클러스터 파생변수 추가
3. 파이캐럿 AutoML을 돌려서 상위 3개 모델을 선정
4. catBoost를 블렌더 모델로 선정하여 전방 모델은 2번 상위모델 3개로 배치 = 스택킹
5. 학습하여 test.csv를 찍어서 submission.csv를 제출


# 고찰내용
두 변수가 연속형이면 피어슨 상관계수로 선형 관계를 확인하지만, 라벨이 명목형일 때는 단순 상관보다 ‘연관성’을 보는 게 맞다.
피어슨은 선형적인 연속형 관계, 스피어만은 순위 기반의 단조 관계를 보는 개념이라 데이터 형태에 맞게 써야 한다.

연속-연속 → 피어슨
순서형-순서형 → 스피어만
명목형 포함 → 카이제곱 또는 그룹 간 검정(ANOVA 등)

-------------------------------------------------------------------------------------


# README.md

```markdown
# 🎯 고객 지원 필요도 예측 (Support Needs Prediction)

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyCaret](https://img.shields.io/badge/PyCaret-3.3.2-orange)](https://pycaret.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-green)](https://catboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**고객 데이터를 활용한 지원 필요도(0: 낮음, 1: 중간, 2: 높음) 분류 프로젝트**.  
PyCaret AutoML로 상위 모델 선정 → KMeans 클러스터링 파생변수 → CatBoost 스태킹 앙상블 적용.

- **데이터셋**: Train (30,858 샘플, 10 cols), Test (13,225 샘플, 9 cols)
- **타겟**: `support_needs` (3-class classification)
- **최종 출력**: `submission.csv` (Accuracy 기반 최적화)

## 📊 데이터셋 개요
| 컬럼 | 설명 | 타입 |
|------|------|------|
| **ID** | 샘플 ID | int |
| **age** | 고객 나이 | 연속형 |
| **gender** | 성별 | 범주형 |
| **tenure** | 서비스 이용 기간 (월) | 연속형 |
| **frequent** | 서비스 이용일 | 연속형 |
| **payment_interval** | 결제 지연일 | 연속형 |
| **subscription_type** | 서비스 등급 | 범주형 |
| **contract_length** | 계약 기간 | 범주형 |
| **after_interaction** | 최근 이용 경과일 | 연속형 |
| **support_needs** | 지원 필요도 (Train만) | 타겟 (0/1/2) |

## 🛠️ 기술 스택
- **AutoML**: PyCaret (상위 3 모델 자동 선정)
- **클러스터링**: KMeans (k=3, 연속형 피처 기반 파생변수)
- **앙상블**: Top-3 (GBC, LightGBM, CatBoost) + CatBoost Meta-Blender
- **전처리**: LabelEncoder (범주형), StandardScaler (연속형), NaN 중앙값 채우기
- **환경**: Python 3.10, Anaconda (alpaco_new env)

## 🚀 설치 & 환경 설정
1. **Anaconda 환경 생성**:
   ```bash
   conda create -n alpaco_new python=3.10
   conda activate alpaco_new
   ```

2. **의존성 설치** (호환성 위해 joblib 고정):
   ```bash
   pip install "pycaret[full]==3.3.2" "joblib==1.3.2" catboost scikit-learn pandas numpy matplotlib seaborn
   ```

3. **파일 준비**:
   - `train.csv`, `test.csv`, `sample_submission.csv`를 루트 폴더에 배치.

## 📁 파일 구조
```
basic-customs/
├── excute.py              # 메인 실행 스크립트
├── train.csv              # 학습 데이터
├── test.csv               # 예측 데이터
├── sample_submission.csv  # 제출 템플릿
├── submission.csv         # 생성된 예측 결과
├── README.md              # 이 파일
└── requirements.txt       # 의존성 목록
```

**requirements.txt**:
```
pycaret[full]==3.3.2
joblib==1.3.2
catboost==1.2.5
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
```

## 🔄 실행 방법
1. **코드 실행**:
   ```bash
   cd basic-customs
   python excute.py
   ```
   - **소요 시간**: 10~15분 (PyCaret AutoML + 학습).
   - **GPU 지원**: CatBoost `task_type='GPU'` 추가 (CUDA 필요).

2. **출력**:
   - `submission.csv` 자동 생성.
   - 콘솔: 전처리 로그, PyCaret 비교 테이블, 클래스 분포.

3. **트러블슈팅**:
   - **ImportError (joblib)**: `pip install joblib==1.3.2 --force-reinstall`
   - **String to Float 에러**: 자동 LabelEncode로 해결.
   - **PyCaret silent 에러**: `verbose=False` 사용.

## 🤖 모델링 과정
1. **전처리**:
   - 범주형 (`gender`, `subscription_type`, `contract_length`): LabelEncoder.
   - 연속형 (`age`, `tenure`, ...): `pd.to_numeric` + 중앙값 NaN Fill.
   - **KMeans**: 연속형 스케일링 → k=3 클러스터 파생변수 추가.

2. **AutoML**:
   - PyCaret `setup` → `compare_models(n_select=3, sort='Accuracy')`.
   - 상위 3: **GBC (0.5307)**, **LightGBM (0.5300)**, **CatBoost (0.5123)**.

3. **스태킹**:
   - Level-1: Top-3 모델 predict_proba → 메타 피처 (9차원).
   - Level-2: CatBoost (iterations=300) 학습.
   - 최종 예측: `np.argmax(proba, axis=1)`.

## 📈 실행 결과 & 성능
### PyCaret 모델 비교 (CV Accuracy 기준)
| Model                  | Accuracy | AUC    | Recall | Prec. | F1     | Kappa  | TT (Sec) |
|-----------------------|----------|--------|--------|-------|--------|--------|----------|
| **gbc**              | **0.5307** | 0.0000 | 0.5307 | 0.4621 | 0.4615 | 0.2366 | 1.461   |
| **lightgbm**         | **0.5300** | 0.6808 | 0.5300 | 0.4928 | 0.4786 | 0.2448 | 0.505   |
| **catboost**         | **0.5123** | 0.6704 | 0.5123 | 0.4793 | 0.4821 | 0.2139 | 3.621   |
| xgboost              | 0.5076  | 0.6662 | 0.5076 | 0.4818 | 0.4854 | 0.2089 | 0.167   |
| rf                   | 0.5066  | 0.6657 | 0.5066 | 0.4778 | 0.4821 | 0.2090 | 1.085   |
| ... (전체 16개)      | ...     | ...    | ...    | ...   | ...    | ...    | ...      |

- **상위 3 모델**:
  1. **Gradient Boosting Classifier** (Acc: 0.5307)
  2. **LightGBM** (Acc: 0.5300)
  3. **CatBoost** (Acc: 0.5123)

### 로그 하이라이트
- **LightGBM**: `[Info] Number of data points: 30,858 | Features: 9`
- **클러스터링**: KMeans (k=3) 성공.
- **Submission**: 생성 완료! 클래스 분포 출력.

### Submission 미리보기
```
ID,suuport_needs
1,0
2,1
...
```
- **클래스 분포**: 0/1/2 균형 (실제 출력 기반).

## 📤 제출 & 성능
- **파일**: `submission.csv` → 대회 플랫폼 업로드.
- **예상 LB Score**: CV Acc 0.53+ (스태킹으로 +2~5% 향상).
- **로컬 평가**: Train Acc (과적합 확인) 콘솔 출력.

## 🔮 개선 아이디어
- **KFold OOF**: 엄격한 스태킹 (leakage 방지).
- **하이퍼파라미터**: `tune_model(top3_models[0])`.
- **피처 엔지니어링**: 상관분석 (Spearman), PCA.
- **앙상블 확장**: `blend_models()` + Voting.
- **실루엣 스코어**: KMeans k 최적화.

## 📝 작성자
- **작성자**: 김윤성
- **이슈**: 에러/개선 PR 환영!
