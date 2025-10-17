# 🎯 고객 지원 필요도 예측 (Support Needs Prediction)

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyCaret](https://img.shields.io/badge/PyCaret-3.3.2-orange)](https://pycaret.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-green)](https://catboost.ai/)
[![Scikit-learn](https://img.shields.io/badge/Scikit-learn-1.3-yellow)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**고객 서비스 데이터를 활용한 지원 필요도(0: 낮음, 1: 중간, 2: 높음) 3-class 분류 프로젝트**.  
PyCaret AutoML로 상위 모델 자동 선정 → KMeans 클러스터링 파생변수 추가 → CatBoost 스태킹 앙상블.

- **데이터셋**: Train (30,858 샘플 × 10 컬럼), Test (13,225 샘플 × 9 컬럼)
- **타겟**: `support_needs` (Ordinal: 0/1/2)
- **최종 제출**: `submission.csv` (ID + 예측 클래스)
- **실행 결과**: CV Accuracy 상위 ~0.53, 스태킹 적용

## 📊 데이터셋 개요
| 컬럼              | 설명                          | 타입       |
|-------------------|-------------------------------|------------|
| **ID**           | 샘플 고유 ID                 | 정수      |
| **age**          | 고객 나이                    | 연속형    |
| **gender**       | 고객 성별                    | 범주형    |
| **tenure**       | 서비스 이용 총 기간 (월)     | 연속형    |
| **frequent**     | 서비스 이용일                | 연속형    |
| **payment_interval** | 결제 지연일               | 연속형    |
| **subscription_type** | 서비스 등급              | 범주형    |
| **contract_length** | 계약 기간                  | 범주형    |
| **after_interaction** | 최근 서비스 이용 경과 (일) | 연속형    |
| **support_needs**| 지원 필요도 (Train만)        | 타겟 (0/1/2) |

## 🛠️ 기술 스택 & 접근법
- **전처리**: LabelEncoder (`gender`, `subscription_type`), StandardScaler (연속형 6개 피처)
- **파생변수**: KMeans 클러스터링 (k=3, 연속형 피처 기반)
- **AutoML**: PyCaret (`compare_models(n_select=3, sort='Accuracy')`)
- **앙상블**: Top-3 모델 predict_proba → CatBoost Meta-Classifier (iterations=300)
- **환경**: Python 3.10+, Anaconda (`alpaco_new`)

## 🚀 설치 & 실행 가이드
### 1. 환경 설정
```bash
# Anaconda 환경 생성 & 활성화
conda create -n alpaco_new python=3.10
conda activate alpaco_new

# 필수 패키지 설치 (호환성 주의: joblib 1.3.2 고정)
pip install "pycaret[full]==3.3.2" "joblib==1.3.2" catboost scikit-learn pandas numpy
2. 파일 준비
train.csv, test.csv, sample_submission.csv를 프로젝트 루트에 배치.
excute.py (아래 코드) 저장.
3. 실행
Bash

cd your-project-folder
python excute.py
소요 시간: ~10-15분 (PyCaret CV + 학습).
출력 로그: 전처리 확인 → KMeans 완료 → PyCaret 테이블 → submission 생성.
GPU 가속: CatBoost에 task_type='GPU' 추가 (CUDA 필요).
주요 코드 (excute.py)
Python

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from pycaret.classification import setup, compare_models
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드 & 전처리
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

X = train.drop(columns=['ID', 'support_needs'])
y = train['support_needs']
X_test = test.drop(columns=['ID'])

# Label Encoding
le_gender = LabelEncoder()
X['gender'] = le_gender.fit_transform(X['gender'])
X_test['gender'] = le_gender.transform(X_test['gender'])

le_sub = LabelEncoder()
X['subscription_type'] = le_sub.fit_transform(X['subscription_type'])
X_test['subscription_type'] = le_sub.transform(X_test['subscription_type'])

# 연속형 스케일링 & KMeans
continuous_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length', 'after_interaction']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[continuous_features])
X_test_scaled = scaler.transform(X_test[continuous_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
X['cluster'] = kmeans.fit_predict(X_scaled)
X_test['cluster'] = kmeans.predict(X_test_scaled)

# PyCaret AutoML
py_train = X.copy()
py_train['support_needs'] = y
setup(data=py_train, target='support_needs', session_id=42, normalize=False, verbose=False)
top3_models = compare_models(sort='Accuracy', n_select=3)

# 스태킹
train_preds = [m.fit(X, y).predict_proba(X) for m in top3_models]
test_preds = [m.predict_proba(X_test) for m in top3_models]
X_meta_train, X_meta_test = np.column_stack(train_preds), np.column_stack(test_preds)

cat_meta = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05, verbose=0, random_seed=42)
cat_meta.fit(X_meta_train, y)
final_pred = cat_meta.predict(X_meta_test)

# 제출
submission['support_needs'] = final_pred.astype(int)
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv 생성 완료!")
📈 성능 결과 (PyCaret CV)
상위 모델 랭킹
Rank	Model	Accuracy	AUC	Recall	Prec.	F1	Kappa	TT (Sec)
1	Gradient Boosting Classifier	0.5307	0.0000	0.5307	0.4621	0.4615	0.2366	1.461
2	LightGBM	0.5300	0.6808	0.5300	0.4928	0.4786	0.2448	0.505
3	CatBoost Classifier	0.5123	0.6704	0.5123	0.4793	0.4821	0.2139	3.621
전체 비교: 16개 모델 테스트 (PyCaret 내부 CV).
LightGBM 로그 예시: [Info] Data points: 30,858 | Features: 9 | Bins: 215
스태킹 효과: Meta-CatBoost로 최종 예측 향상 예상 (+2-5%).
콘솔 출력 하이라이트
text

Train shape: (30858, 10)
Test shape: (13225, 9)
✔ KMeans 완료
[PyCaret Processing: 100%]
=== 상위 3개 모델 ===
1. GradientBoostingClassifier(...)
2. LGBMClassifier(...)
3. CatBoostClassifier(...)
✅ submission.csv 생성 완료!
📁 프로젝트 구조
text

project/
├── excute.py                 # 메인 스크립트
├── train.csv                 # 학습 데이터
├── test.csv                  # 테스트 데이터
├── sample_submission.csv     # 템플릿
├── submission.csv            # 예측 결과 (생성됨)
├── README.md                 # 이 문서
└── requirements.txt          # pip install -r
requirements.txt 예시:

text

pycaret[full]==3.3.2
catboost==1.2.5
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
joblib==1.3.2
🔮 개선 제안
피처 엔지니어링: Spearman 상관분석 (명목형 타겟), PCA.
KFold OOF: Leakage 방지 스태킹 강화.
튜닝: tune_model(top3_models[0]) 또는 Optuna.
클러스터 최적화: Elbow/Silhouette로 k 결정.
앙상블 확장: VotingClassifier 또는 PyCaret stack_models().
📤 제출 & 평가
파일 업로드: submission.csv를 대회 사이트에 제출.
클래스 분포: 균형 (실행 시 submission['support_needs'].value_counts() 확인).
로컬 검증: Train 예측으로 classification_report 출력 추가 가능.
🙌 기여 & 연락
Issues/PR: 버그 수정, 기능 추가 환영!
작성자: [Your GitHub]
라이선스: MIT
⭐ Star 주시고 Fork 해보세요! Questions? Issues 열기

참고 자료
PyCaret Documentation
CatBoost Guide
Kaggle/Dacon Submission Tips
text


---

### 📝 **사용 지침**
1. **GitHub Repo 생성**: README.md 붙여넣기 → Commit.
2. **커스터마이징**:
   - Badges 링크: 실제 Repo URL로.
   - 성능 테이블: 실제 실행 결과로 업데이트.
   - 이미지: PyCaret 플롯 스크린샷 추가 (`![Leaderboard](leaderboard.png)`).
3. **requirements.txt 생성**:
   ```bash
   pip freeze > requirements.txt
트러블슈팅 섹션 추가: 이전 에러 (joblib, silent) 언급.