import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from pycaret.classification import setup, compare_models
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터 로드
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# 2. 전처리
X = train.drop(columns=['ID', 'support_needs'])
y = train['support_needs']
X_test = test.drop(columns=['ID'])

# 문자열 컬럼 인코딩
le_gender = LabelEncoder()
X['gender'] = le_gender.fit_transform(X['gender'])
X_test['gender'] = le_gender.transform(X_test['gender'])

le_sub = LabelEncoder()
X['subscription_type'] = le_sub.fit_transform(X['subscription_type'])
X_test['subscription_type'] = le_sub.transform(X_test['subscription_type'])

# 수치형 변수 리스트
continuous_features = [
    'age', 'tenure', 'frequent',
    'payment_interval', 'contract_length', 'after_interaction'
]

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[continuous_features])
X_test_scaled = scaler.transform(X_test[continuous_features])

# 3. KMeans 클러스터 생성
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
X['cluster'] = kmeans.fit_predict(X_scaled)
X_test['cluster'] = kmeans.predict(X_test_scaled)

print("\n✔ KMeans 완료")

# 4. PyCaret AutoML
py_train = X.copy()
py_train['support_needs'] = y

setup(
    data=py_train,
    target='support_needs',
    session_id=42,
    normalize=False,
    verbose=False   # silent 제거
)

top3_models = compare_models(sort='Accuracy', n_select=3)
print("\n=== 상위 3개 모델 ===")
for i, m in enumerate(top3_models, 1):
    print(f"{i}. {m}")

# 5. 스태킹 준비
test_preds = []
for model in top3_models:
    model.fit(X, y)
    test_preds.append(model.predict_proba(X_test))

train_preds = [m.predict_proba(X) for m in top3_models]
X_meta_train = np.column_stack(train_preds)
X_meta_test = np.column_stack(test_preds)

# 6. CatBoost 메타모델
cat_meta = CatBoostClassifier(
    iterations=300,
    depth=5,
    learning_rate=0.05,
    loss_function='MultiClass',
    random_seed=42,
    verbose=0
)
cat_meta.fit(X_meta_train, y)
final_pred = cat_meta.predict(X_meta_test)

# 7. 제출
submission['support_needs'] = final_pred.astype(int)
submission.to_csv('submission.csv', index=False)
print("\n✅ submission.csv 생성 완료!")