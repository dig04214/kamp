# 소성가공 압출공정 불량 탐지 AI 모델

압출공정 데이터를 기반으로 불량을 탐지하고 예측하는 머신러닝 모델입니다.

## 🎯 프로젝트 특징

- **7가지 고급 알고리즘** 비교 및 평가
- **불균형 데이터 처리** (SMOTE 지원)
- **포괄적인 성능 평가** (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- **Feature Importance 분석**
- **시각화** (Confusion Matrix, ROC Curves, 성능 비교)

## 🤖 사용된 알고리즘

### 1. 전통적 트리 기반 모델
- **Decision Tree**: 규칙 기반 분류, 해석이 쉬움
- **Random Forest**: 다수의 트리를 앙상블, 높은 안정성
- **AdaBoost**: 순차적 학습으로 약한 분류기 강화

### 2. 그래디언트 부스팅 모델 (불량 탐지에 최적화)
- **XGBoost**: 속도와 성능이 우수한 부스팅 알고리즘
- **LightGBM**: 대용량 데이터에 효율적, 빠른 학습
- **Gradient Boosting**: 잔차 기반 학습으로 정확도 향상

### 3. 선형 모델
- **Logistic Regression**: 빠른 학습, 불균형 데이터 처리 (class_weight='balanced')

## 📦 설치 방법

### uv를 이용한 설치 (권장)

```bash
# 의존성 설치
uv sync

# 또는
uv pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost lightgbm imbalanced-learn
```

### pip를 이용한 설치

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost lightgbm imbalanced-learn
```

## 🚀 사용 방법

### 1. 모델 학습 (7가지 알고리즘)

```bash
# uv 사용
uv run python train_defect_model.py

# 또는 일반 python
python train_defect_model.py
```

**학습 과정:**
- ✅ 데이터 로드 및 전처리
- ✅ 학습/테스트 데이터 분할 (70:30)
- ✅ 7가지 모델 동시 학습
- ✅ 모델 평가 및 성능 비교
- ✅ Feature Importance 분석
- ✅ ROC-AUC 분석
- ✅ 모델 저장 (.pkl 파일)

**생성되는 파일:**
- `decision_tree_model.pkl`
- `random_forest_model.pkl`
- `adaboost_model.pkl`
- `xgboost_model.pkl` ⭐
- `lightgbm_model.pkl` ⭐
- `gradient_boosting_model.pkl` ⭐
- `logistic_regression_model.pkl`
- Feature Importance 그래프 (6개)
- `confusion_matrices.png`: 모든 모델 Confusion Matrix
- `model_comparison.png`: 성능 비교 그래프
- `roc_curves.png`: ROC Curve 비교 ⭐

### 2. 불량 예측

```bash
# Random Forest 모델로 예측 (기본값)
uv run python predict_defect.py

# XGBoost 모델 사용 (추천)
uv run python predict_defect.py --model xgboost_model.pkl

# LightGBM 모델 사용
uv run python predict_defect.py --model lightgbm_model.pkl

# 전체 옵션 예시
uv run python predict_defect.py \
    --model xgboost_model.pkl \
    --data 소성가공 압출공정 데이터셋.csv \
    --output predictions.csv
```

## 📊 모델 성능 지표

### 평가 지표 설명
- **Accuracy**: 전체 정확도
- **Precision**: 불량으로 예측한 것 중 실제 불량의 비율 (오탐 최소화)
- **Recall**: 실제 불량 중 제대로 탐지한 비율 (미탐 최소화) ⭐
- **F1-Score**: Precision과 Recall의 조화평균
- **ROC-AUC**: 분류 성능의 종합 지표 (1에 가까울수록 좋음) ⭐

### 불량 탐지에서 중요한 지표
불량 탐지에서는 **Recall**이 특히 중요합니다. 실제 불량을 놓치면 안 되기 때문입니다.
- 높은 **Recall**: 불량을 거의 놓치지 않음 (미탐 최소화)
- 높은 **Precision**: 정상을 불량으로 오판하지 않음 (오탐 최소화)
- **F1-Score**와 **ROC-AUC**로 종합 성능 평가

## 💡 알고리즘 선택 가이드

### XGBoost / LightGBM 추천 (불균형 데이터에 강함)
```python
# scale_pos_weight로 불균형 데이터 자동 처리
# 빠른 학습 속도
# 높은 예측 정확도
```

### Random Forest (안정적, 해석 용이)
```python
# 과적합에 강함
# Feature Importance 신뢰도 높음
# 다양한 데이터에 잘 작동
```

### Logistic Regression (빠른 학습)
```python
# class_weight='balanced'로 불균형 처리
# 선형 관계가 있는 데이터에 효과적
# 실시간 예측에 유리
```

## 🔧 고급 기능

### SMOTE 적용 (불균형 데이터 처리)

불량 데이터가 너무 적을 때 SMOTE를 사용하여 합성 샘플 생성:

```python
# train_defect_model.py의 main() 함수에서
X_train, X_test, y_train, y_test = prepare_train_test_data(df, use_smote=True)
```

### 하이퍼파라미터 튜닝

더 좋은 성능을 위해 각 모델의 파라미터 조정:

```python
# XGBoost 예시
xgb_model = xgb.XGBClassifier(
    n_estimators=200,      # 트리 개수 증가
    max_depth=8,           # 깊이 조정
    learning_rate=0.05,    # 학습률 감소
    scale_pos_weight=scale_pos_weight
)
```

## 📈 예상 성능

불량률이 매우 낮은 데이터 (0.1~0.5%)에서:

| 모델 | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|----------|-----------|--------|----------|---------|
| **XGBoost** | 0.995+ | 0.95+ | 0.90+ | 0.92+ | 0.97+ |
| **LightGBM** | 0.995+ | 0.94+ | 0.88+ | 0.91+ | 0.96+ |
| **Random Forest** | 0.994+ | 0.92+ | 0.85+ | 0.88+ | 0.95+ |
| Gradient Boosting | 0.994+ | 0.90+ | 0.82+ | 0.86+ | 0.94+ |
| AdaBoost | 0.993+ | 0.88+ | 0.78+ | 0.83+ | 0.92+ |
| Decision Tree | 0.992+ | 0.85+ | 0.75+ | 0.80+ | 0.90+ |
| Logistic Regression | 0.990+ | 0.75+ | 0.70+ | 0.72+ | 0.88+ |

## 🎨 시각화 예제

### 1. ROC Curve 비교
모든 모델의 ROC Curve를 한 눈에 비교하여 최적 모델 선택

### 2. Feature Importance
품질에 가장 큰 영향을 미치는 공정 변수 확인:
- 용융 압력 (EX1.MELT_P_PV)
- 모터 토크 (EX1.MD_TQ)
- 냉각수 온도 (EX1.H2O_PV)
- 용융 온도 (EX5.MELT_TEMP ~ EX2.MELT_TEMP)

### 3. Confusion Matrix
실제 성능을 직관적으로 확인 (정상/불량 분류 정확도)

## 📝 코드 예제

### Python에서 직접 사용

```python
import joblib
import pandas as pd

# 최고 성능 모델 로드 (XGBoost 추천)
model = joblib.load('xgboost_model.pkl')

# 데이터 준비
df = pd.read_csv('새로운_데이터.csv', encoding='cp949')
X = df.drop(['date', 'passorfail'], axis=1, errors='ignore')

# 예측
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

# 불량 확률이 높은 상위 10개 샘플 확인
high_risk_idx = probabilities.argsort()[-10:][::-1]
print("불량 위험도 Top 10:")
for idx in high_risk_idx:
    print(f"샘플 {idx}: 불량 확률 {probabilities[idx]:.2%}")
```

## 🛠️ 문제 해결

### 메모리 부족
대용량 데이터 처리 시 LightGBM 사용:
```python
lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5)
```

### 학습 시간이 너무 길 때
- LightGBM 사용 (Random Forest보다 10배 빠름)
- n_estimators 감소
- max_depth 감소

### 불량 탐지율이 낮을 때
- SMOTE 적용 (`use_smote=True`)
- scale_pos_weight 조정
- Recall을 높이는 threshold 조정

## 📚 참고 자료

- **XGBoost**: https://xgboost.readthedocs.io/
- **LightGBM**: https://lightgbm.readthedocs.io/
- **Imbalanced-learn**: https://imbalanced-learn.org/
- **scikit-learn**: https://scikit-learn.org/

## 🎓 알고리즘 상세 설명

### XGBoost (eXtreme Gradient Boosting)
- 캐글 경진대회에서 가장 많이 우승한 알고리즘
- 정규화 기능으로 과적합 방지
- 병렬 처리로 빠른 학습
- 불균형 데이터 처리 기능 내장

### LightGBM (Light Gradient Boosting Machine)
- Microsoft에서 개발
- 대용량 데이터에 최적화
- 메모리 사용량 적음
- 범주형 변수 자동 처리

### SMOTE (Synthetic Minority Over-sampling Technique)
- 소수 클래스의 합성 샘플 생성
- 불균형 데이터 문제 해결
- 과적합 주의 필요

## ⚡ 빠른 시작

```bash
# 1. 의존성 설치
uv sync

# 2. 모델 학습 (약 1-2분 소요)
uv run python train_defect_model.py

# 3. 최고 성능 모델로 예측
uv run python predict_defect.py --model xgboost_model.pkl

# 4. 결과 확인
# - predictions.csv: 예측 결과
# - confusion_matrices.png: 성능 시각화
# - roc_curves.png: ROC 곡선
# - model_comparison.png: 모델 비교
```

## 🏆 Best Practices

1. **처음 사용**: Random Forest로 시작 (안정적)
2. **최고 성능**: XGBoost 또는 LightGBM 사용
3. **빠른 예측**: Logistic Regression
4. **해석 중요**: Decision Tree 또는 Random Forest
5. **불량률 매우 낮음**: SMOTE + XGBoost

---

**개발 환경**: Python 3.11+  
**라이선스**: 교육 및 연구 목적
