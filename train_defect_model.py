"""
소성가공 압출공정 불량 탐지 모델 학습
Decision Tree, Random Forest, AdaBoost, XGBoost, LightGBM, GradientBoosting 모델을 사용한 불량 예측
불균형 데이터 처리를 위한 SMOTE 적용
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess_data(file_path):
    """데이터 로드 및 전처리"""
    print("=" * 50)
    print("데이터 로딩 중...")
    print("=" * 50)
    
    # CSV 파일 로드
    df = pd.read_csv(file_path, encoding='cp949')
    
    print(f"\n데이터셋 크기: {df.shape}")
    print(f"컬럼 목록: {df.columns.tolist()}")
    
    # date 컬럼 제거 
    if 'date' in df.columns:
        df = df.drop('date', axis=1)
    
    # 결측치 확인
    print(f"\n결측치 개수:\n{df.isnull().sum()}")
    
    # 결측치가 있는 행 제거
    df = df.dropna()
    
    # passorfail 컬럼의 분포 확인
    print("\n불량 분포:")
    print(df['passorfail'].value_counts())
    print(f"불량률: {df['passorfail'].sum() / len(df) * 100:.2f}%")
    
    return df


def prepare_train_test_data(df, test_size=0.3, random_state=42, use_smote=False):
    """학습/테스트 데이터 분할"""
    print("\n" + "=" * 50)
    print("데이터 분할 중...")
    print("=" * 50)
    
    # Feature와 Target 분리
    X = df.drop('passorfail', axis=1)
    y = df['passorfail']
    
    # 학습/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n학습 데이터 크기: {X_train.shape}")
    print(f"테스트 데이터 크기: {X_test.shape}")
    print(f"\n학습 데이터 불량 분포:\n{y_train.value_counts()}")
    print(f"테스트 데이터 불량 분포:\n{y_test.value_counts()}")
    
    # SMOTE 적용 (불균형 데이터 처리)
    if use_smote:
        print("\n" + "=" * 50)
        print("SMOTE 적용 중 (불균형 데이터 처리)...")
        print("=" * 50)
        smote = SMOTE(random_state=random_state)
        resampled = smote.fit_resample(X_train, y_train)
        X_train, y_train = resampled[0], resampled[1]
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=X.columns)

        # Ensure y_train is a 1-D Series
        if not isinstance(y_train, pd.Series):
            y_arr = np.asarray(y_train).ravel()
            # use original target name if available
            target_name = getattr(y, 'name', 'target')
            y_train = pd.Series(y_arr, name=target_name)
        print(f"\nSMOTE 적용 후 학습 데이터 크기: {X_train.shape}")
        print(f"SMOTE 적용 후 불량 분포:\n{pd.Series(y_train).value_counts()}")
    
    return X_train, X_test, y_train, y_test


def train_decision_tree(X_train, y_train, max_depth=10, random_state=42):
    """Decision Tree 모델 학습"""
    print("\n" + "=" * 50)
    print("Decision Tree 모델 학습 중...")
    print("=" * 50)
    
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X_train, y_train)
    
    print("Decision Tree 학습 완료!")
    return dt


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """Random Forest 모델 학습"""
    print("\n" + "=" * 50)
    print("Random Forest 모델 학습 중...")
    print("=" * 50)
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    print("Random Forest 학습 완료!")
    return rf


def train_adaboost(X_train, y_train, n_estimators=50, random_state=42):
    """AdaBoost 모델 학습"""
    print("\n" + "=" * 50)
    print("AdaBoost 모델 학습 중...")
    print("=" * 50)
    
    ada = AdaBoostClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        algorithm='SAMME'
    )
    ada.fit(X_train, y_train)
    
    print("AdaBoost 학습 완료!")
    return ada


def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, random_state=42):
    """XGBoost 모델 학습"""
    print("\n" + "=" * 50)
    print("XGBoost 모델 학습 중...")
    print("=" * 50)
    
    # 클래스 불균형 처리를 위한 scale_pos_weight 계산
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    
    print("XGBoost 학습 완료!")
    return xgb_model


def train_lightgbm(X_train, y_train, n_estimators=100, max_depth=6, random_state=42):
    """LightGBM 모델 학습"""
    print("\n" + "=" * 50)
    print("LightGBM 모델 학습 중...")
    print("=" * 50)
    
    # 클래스 불균형 처리를 위한 scale_pos_weight 계산
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    print("LightGBM 학습 완료!")
    return lgb_model


def train_gradient_boosting(X_train, y_train, n_estimators=100, max_depth=5, random_state=42):
    """Gradient Boosting 모델 학습"""
    print("\n" + "=" * 50)
    print("Gradient Boosting 모델 학습 중...")
    print("=" * 50)
    
    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        random_state=random_state
    )
    gb.fit(X_train, y_train)
    
    print("Gradient Boosting 학습 완료!")
    return gb


def train_logistic_regression(X_train, y_train, random_state=42):
    """Logistic Regression 모델 학습"""
    print("\n" + "=" * 50)
    print("Logistic Regression 모델 학습 중...")
    print("=" * 50)
    
    # 클래스 불균형 처리
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    
    lr = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced'
    )
    lr.fit(X_train, y_train)
    
    print("Logistic Regression 학습 완료!")
    return lr


def evaluate_model(model, X_test, y_test, model_name):
    """모델 평가"""
    print("\n" + "=" * 50)
    print(f"{model_name} 모델 평가")
    print("=" * 50)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 확률 예측 (ROC-AUC 계산용)
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
    else:
        y_pred_proba = None
        roc_auc = 0
    
    # 평가 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred, target_names=['정상', '불량']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_pred_proba': y_pred_proba
    }


def plot_feature_importance(model, feature_names, model_name, top_n=15):
    """Feature Importance 시각화"""
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        importances.head(top_n).plot(kind='barh')
        plt.title(f'{model_name} - Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(f'{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"\n{model_name} Feature Importance 저장 완료: {model_name}_feature_importance.png")
        plt.close()
        
        return importances


def plot_confusion_matrices(results):
    """모든 모델의 Confusion Matrix 시각화"""
    n_models = len(results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, result in enumerate(results):
        sns.heatmap(
            result['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=axes[idx],
            xticklabels=['정상', '불량'],
            yticklabels=['정상', '불량']
        )
        axes[idx].set_title(f"{result['model_name']}\nF1: {result['f1_score']:.4f} | ROC-AUC: {result['roc_auc']:.4f}")
        axes[idx].set_ylabel('실제')
        axes[idx].set_xlabel('예측')
    
    # 빈 서브플롯 제거
    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("\nConfusion Matrices 저장 완료: confusion_matrices.png")
    plt.close()


def compare_models(results):
    """모델 성능 비교"""
    print("\n" + "=" * 50)
    print("모델 성능 비교")
    print("=" * 50)
    
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Accuracy': r['accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1-Score': r['f1_score'],
            'ROC-AUC': r['roc_auc']
        }
        for r in results
    ])
    
    # F1-Score로 정렬
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
    
    print("\n", comparison_df.to_string(index=False))
    
    # 성능 비교 그래프
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    cmap = plt.cm.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(comparison_df))]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(range(len(comparison_df)), comparison_df[metric], color=colors)
        ax.set_title(f'{metric} 비교', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_xticks(range(len(comparison_df)))
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        # 값 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
    
    # 종합 랭킹 표시
    ax = axes[1, 2]
    ranking_data = comparison_df[['Model', 'F1-Score', 'ROC-AUC']].head(5)
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=ranking_data.values,
                     colLabels=ranking_data.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title('Top 5 Models (F1-Score 기준)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n모델 비교 그래프 저장 완료: model_comparison.png")
    plt.close()
    
    return comparison_df


def plot_roc_curves(results, y_test):
    """ROC Curve 시각화"""
    print("\nROC Curve 생성 중...")
    
    plt.figure(figsize=(12, 8))
    
    for result in results:
        if result['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {result['roc_auc']:.4f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - 모델 비교', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    print("ROC Curves 저장 완료: roc_curves.png")
    plt.close()


def save_models(models_dict):
    """학습된 모델 저장"""
    print("\n" + "=" * 50)
    print("모델 저장 중...")
    print("=" * 50)
    
    saved_files = []
    for name, model in models_dict.items():
        filename = f"{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, filename)
        saved_files.append(filename)
        print(f"  - {filename}")
    
    print("\n모델 저장 완료!")
    return saved_files


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 50)
    print("소성가공 압출공정 불량 탐지 모델 학습 시작")
    print("=" * 50)
    
    # 1. 데이터 로드 및 전처리
    df = load_and_preprocess_data('소성가공 압출공정 데이터셋.csv')
    
    # 2. 학습/테스트 데이터 분할 (SMOTE 적용 여부 선택 가능)
    X_train, X_test, y_train, y_test = prepare_train_test_data(df, use_smote=True)
    
    # 3. 여러 모델 학습
    print("\n" + "=" * 50)
    print("다양한 알고리즘으로 모델 학습 중...")
    print("=" * 50)
    
    models = {}
    models['Decision Tree'] = train_decision_tree(X_train, y_train)
    models['Random Forest'] = train_random_forest(X_train, y_train)
    models['AdaBoost'] = train_adaboost(X_train, y_train)
    models['XGBoost'] = train_xgboost(X_train, y_train)
    models['LightGBM'] = train_lightgbm(X_train, y_train)
    models['Gradient Boosting'] = train_gradient_boosting(X_train, y_train)
    models['Logistic Regression'] = train_logistic_regression(X_train, y_train)
    
    # 4. 모델 평가
    results = []
    for name, model in models.items():
        results.append(evaluate_model(model, X_test, y_test, name))
    
    # 5. Feature Importance 분석 (트리 기반 모델만)
    feature_names = X_train.columns
    for name, model in models.items():
        if name in ['Decision Tree', 'Random Forest', 'AdaBoost', 'XGBoost', 'LightGBM', 'Gradient Boosting']:
            plot_feature_importance(model, feature_names, name)
    
    # 6. Confusion Matrix 시각화
    plot_confusion_matrices(results)
    
    # 7. ROC Curve 시각화
    plot_roc_curves(results, y_test)
    
    # 8. 모델 성능 비교
    comparison_df = compare_models(results)
    
    # 9. 모델 저장
    saved_files = save_models(models)
    
    # 10. 최종 결과 출력
    print("\n" + "=" * 50)
    print("학습 완료!")
    print("=" * 50)
    print("\n생성된 모델 파일:")
    for file in saved_files:
        print(f"  - {file}")
    
    print("\n생성된 시각화 파일:")
    print("  - confusion_matrices.png")
    print("  - model_comparison.png")
    print("  - roc_curves.png")
    for name in ['Decision Tree', 'Random Forest', 'AdaBoost', 'XGBoost', 'LightGBM', 'Gradient Boosting']:
        print(f"  - {name}_feature_importance.png")
    
    # 최고 성능 모델 출력 (F1-Score 기준)
    best_idx = comparison_df['F1-Score'].idxmax()
    best_model = comparison_df.loc[best_idx]
    print(f"\n{'='*50}")
    print("🏆 최고 성능 모델 (F1-Score 기준)")
    print(f"{'='*50}")
    print(f"모델: {best_model['Model']}")
    print(f"  - Accuracy:  {best_model['Accuracy']:.4f}")
    print(f"  - Precision: {best_model['Precision']:.4f}")
    print(f"  - Recall:    {best_model['Recall']:.4f}")
    print(f"  - F1-Score:  {best_model['F1-Score']:.4f}")
    print(f"  - ROC-AUC:   {best_model['ROC-AUC']:.4f}")
    
    # Top 3 모델 출력
    print(f"\n{'='*50}")
    print("📊 Top 3 모델")
    print(f"{'='*50}")
    for i, (idx, row) in enumerate(comparison_df.head(3).iterrows(), 1):
        print(f"\n{i}. {row['Model']}")
        print(f"   F1-Score: {row['F1-Score']:.4f} | ROC-AUC: {row['ROC-AUC']:.4f}")


if __name__ == "__main__":
    main()
