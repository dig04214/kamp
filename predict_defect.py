"""
학습된 모델을 사용한 불량 예측 스크립트
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path


def load_model(model_path):
    """저장된 모델 로드"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    print(f"모델 로딩 중: {model_path}")
    model = joblib.load(model_path)
    print("모델 로딩 완료!")
    return model


def load_data(file_path):
    """예측할 데이터 로드"""
    print(f"\n데이터 로딩 중: {file_path}")
    df = pd.read_csv(file_path, encoding='cp949')
    
    # date 컬럼이 있으면 제거
    if 'date' in df.columns:
        dates = df['date'].copy()
        df = df.drop('date', axis=1)
    else:
        dates = None
    
    # passorfail 컬럼이 있으면 제거 (예측 데이터이므로)
    has_label = 'passorfail' in df.columns
    if has_label:
        true_labels = df['passorfail'].copy()
        df = df.drop('passorfail', axis=1)
    else:
        true_labels = None
    
    # 결측치 확인
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"결측치 발견: {null_count}개")
        print("결측치가 있는 행 제거 중...")
        # 결측치가 있는 행의 인덱스 저장
        valid_idx = df.dropna().index
        df = df.loc[valid_idx]
        if dates is not None:
            dates = dates.loc[valid_idx]
        if true_labels is not None:
            true_labels = true_labels.loc[valid_idx]
        print(f"결측치 제거 후 데이터 크기: {df.shape}")
    
    print(f"데이터 크기: {df.shape}")
    return df, dates, true_labels, has_label


def predict(model, X, threshold=0.5):
    """불량 예측"""
    print("\n예측 수행 중...")
    
    # 확률 예측 (가능한 경우)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        defect_probs = probabilities[:, 1]  # 불량일 확률
        
        # 사용자 지정 임계값으로 예측
        if threshold != 0.5:
            print(f"사용자 지정 임계값 적용: {threshold}")
            predictions = (defect_probs >= threshold).astype(int)
        else:
            predictions = model.predict(X)
    else:
        predictions = model.predict(X)
        defect_probs = None
    
    print("예측 완료!")
    return predictions, defect_probs


def save_predictions(predictions, probs, dates, true_labels, output_path):
    """예측 결과 저장"""
    result_df = pd.DataFrame()
    
    if dates is not None:
        result_df['date'] = dates
    
    result_df['prediction'] = predictions
    result_df['prediction_label'] = result_df['prediction'].map({0: '정상', 1: '불량'})
    
    if probs is not None:
        result_df['defect_probability'] = probs
    
    if true_labels is not None:
        result_df['true_label'] = true_labels
        result_df['true_label_name'] = result_df['true_label'].map({0: '정상', 1: '불량'})
        result_df['correct'] = (result_df['prediction'] == result_df['true_label'])
    
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n예측 결과 저장 완료: {output_path}")
    
    return result_df


def print_prediction_summary(predictions, true_labels=None, probs=None):
    """예측 결과 요약 출력"""
    print("\n" + "=" * 50)
    print("예측 결과 요약")
    print("=" * 50)
    
    total = len(predictions)
    defect_count = np.sum(predictions == 1)
    normal_count = np.sum(predictions == 0)
    
    print(f"\n총 예측 샘플 수: {total}")
    print(f"정상 예측: {normal_count} ({normal_count/total*100:.2f}%)")
    print(f"불량 예측: {defect_count} ({defect_count/total*100:.2f}%)")
    
    # 확률 분포 정보
    if probs is not None:
        print(f"\n불량 확률 통계:")
        print(f"  - 평균: {probs.mean():.4f}")
        print(f"  - 최대: {probs.max():.4f}")
        print(f"  - 최소: {probs.min():.4f}")
        
        # 위험도별 분류
        very_high_risk = np.sum(probs >= 0.9)
        high_risk = np.sum((probs >= 0.8) & (probs < 0.9))
        medium_risk = np.sum((probs >= 0.5) & (probs < 0.8))
        low_risk = np.sum(probs < 0.5)
        
        print(f"\n위험도별 분포:")
        print(f"  - 매우 높음 (≥90%): {very_high_risk}개 ({very_high_risk/total*100:.2f}%)")
        print(f"  - 높음 (80~90%):   {high_risk}개 ({high_risk/total*100:.2f}%)")
        print(f"  - 중간 (50~80%):   {medium_risk}개 ({medium_risk/total*100:.2f}%)")
        print(f"  - 낮음 (<50%):     {low_risk}개 ({low_risk/total*100:.2f}%)")
    
    if true_labels is not None:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        # NaN 체크 및 제거
        nan_count = pd.isna(true_labels).sum()
        if nan_count > 0:
            print(f"\n경고: true_labels에 {nan_count}개의 NaN 발견. 해당 행 제외하고 평가합니다.")
            valid_mask = ~pd.isna(true_labels)
            true_labels = true_labels[valid_mask]
            predictions = predictions[valid_mask]
            print(f"평가에 사용된 샘플 수: {len(predictions)}")
        
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        cm = confusion_matrix(true_labels, predictions)
        
        print("\n" + "=" * 50)
        print("실제 레이블과 비교")
        print("=" * 50)
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"              예측_정상  예측_불량")
        print(f"실제_정상    {cm[0][0]:8d}   {cm[0][1]:8d}")
        print(f"실제_불량    {cm[1][0]:8d}   {cm[1][1]:8d}")


def print_high_risk_samples(result_df, top_n=10, threshold=0.8):
    """고위험 샘플 상세 정보 출력 및 CSV 저장"""
    if 'defect_probability' not in result_df.columns:
        return
    
    # 고위험 샘플 필터링
    high_risk = result_df[result_df['defect_probability'] >= threshold].copy()
    
    if len(high_risk) == 0:
        print(f"\n✅ 불량 확률 {threshold*100:.0f}% 이상인 고위험 샘플이 없습니다.")
        return
    
    # 확률 순으로 정렬
    high_risk = high_risk.sort_values('defect_probability', ascending=False)
    
    print("\n" + "=" * 70)
    print(f"⚠️  고위험 샘플 상세 정보 (불량 확률 ≥ {threshold*100:.0f}%)")
    print("=" * 70)
    
    print(f"\n총 고위험 샘플: {len(high_risk)}개")
    print(f"Top {min(top_n, len(high_risk))}개 샘플 상세:")
    print("-" * 70)
    
    # Top N 출력
    for i, (idx, row) in enumerate(high_risk.head(top_n).iterrows(), 1):
        print(f"\n[{i}] 샘플 정보:")
        if 'date' in row and pd.notna(row['date']):
            print(f"    날짜/시간: {row['date']}")
        print(f"    불량 확률: {row['defect_probability']:.2%} {'🔴' if row['defect_probability'] >= 0.95 else '🟠'}")
        print(f"    예측 결과: {row['prediction_label']}")
        if 'true_label_name' in row and pd.notna(row['true_label_name']):
            correct = "✅" if row.get('correct', False) else "❌"
            print(f"    실제 결과: {row['true_label_name']} {correct}")
    
    # 통계 요약
    print("\n" + "-" * 70)
    print("고위험 샘플 통계:")
    print(f"  - 평균 불량 확률: {high_risk['defect_probability'].mean():.2%}")
    print(f"  - 최대 불량 확률: {high_risk['defect_probability'].max():.2%}")
    print(f"  - 최소 불량 확률: {high_risk['defect_probability'].min():.2%}")
    
    # 정확도 (레이블이 있는 경우)
    if 'correct' in high_risk.columns:
        correct_count = high_risk['correct'].sum()
        total_count = len(high_risk)
        accuracy = correct_count / total_count
        print(f"  - 고위험 샘플 예측 정확도: {accuracy:.2%} ({correct_count}/{total_count})")
    
    # 저장된 파일 안내
    high_risk_file = 'high_risk_samples.csv'
    high_risk.to_csv(high_risk_file, index=False, encoding='utf-8-sig')
    print(f"\n📄 고위험 샘플이 별도 파일로 저장되었습니다: {high_risk_file}")
    
    # 콘솔에 전체 리스트 출력 (간단한 형식)
    print("\n" + "=" * 70)
    print("📋 고위험 샘플 전체 리스트 (콘솔 출력)")
    print("=" * 70)
    
    # 출력할 컬럼 선택 및 헤더 출력
    if 'date' in high_risk.columns:
        print(f"\n{'No.':<5} {'날짜/시간':<20} {'불량확률':<10} {'예측':<8} {'실제':<8} {'정확도':<6}")
    else:
        print(f"\n{'No.':<5} {'불량확률':<10} {'예측':<8} {'실제':<8} {'정확도':<6}")
    print("-" * 70)
    
    # 전체 고위험 샘플 출력
    for i, (idx, row) in enumerate(high_risk.iterrows(), 1):
        date_str = str(row['date'])[:20] if 'date' in row and pd.notna(row['date']) else ""
        prob_str = f"{row['defect_probability']:.2%}"
        pred_str = "불량" if row['prediction'] == 1 else "정상"
        
        if 'true_label_name' in row and pd.notna(row['true_label_name']):
            true_str = str(row['true_label_name'])
            correct_str = "✅" if row.get('correct', False) else "❌"
        else:
            true_str = "N/A"
            correct_str = ""
        
        if 'date' in high_risk.columns:
            print(f"{i:<5} {date_str:<20} {prob_str:<10} {pred_str:<8} {true_str:<8} {correct_str:<6}")
        else:
            print(f"{i:<5} {prob_str:<10} {pred_str:<8} {true_str:<8} {correct_str:<6}")
    
    print("=" * 70)


def list_available_models():
    """사용 가능한 모델 파일 목록 출력"""
    import glob
    model_files = glob.glob('*_model.pkl')
    if model_files:
        print("\n사용 가능한 모델:")
        for i, model_file in enumerate(model_files, 1):
            print(f"  {i}. {model_file}")
    return model_files


def main():
    parser = argparse.ArgumentParser(
        description='소성가공 압출공정 불량 예측 스크립트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # Random Forest 모델 사용 (기본값)
  python predict_defect.py
  
  # XGBoost 모델 사용 (추천)
  python predict_defect.py --model xgboost_model.pkl
  
  # LightGBM 모델 사용
  python predict_defect.py --model lightgbm_model.pkl
  
  # 다른 데이터 파일과 출력 파일 지정
  python predict_defect.py --model xgboost_model.pkl --data test.csv --output result.csv
  
지원 모델:
  - xgboost_model.pkl (추천: 높은 성능)
  - lightgbm_model.pkl (추천: 빠른 속도)
  - random_forest_model.pkl (안정적)
  - gradient_boosting_model.pkl
  - adaboost_model.pkl
  - decision_tree_model.pkl
  - logistic_regression_model.pkl
        """
    )
    parser.add_argument('--model', type=str, default='random_forest_model.pkl',
                        help='모델 파일 경로 (기본값: random_forest_model.pkl)')
    parser.add_argument('--data', type=str, default='소성가공 압출공정 데이터셋.csv',
                        help='예측할 데이터 파일 경로')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='결과 저장 파일 경로 (기본값: predictions.csv)')
    parser.add_argument('--list-models', action='store_true',
                        help='사용 가능한 모델 목록 출력')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='불량 판정 임계값 (0.0~1.0, 기본값: 0.5)')
    parser.add_argument('--high-risk-threshold', type=float, default=0.8,
                        help='고위험 샘플 판정 임계값 (기본값: 0.8)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='출력할 고위험 샘플 개수 (기본값: 10)')
    
    args = parser.parse_args()
    
    # 모델 목록 출력 후 종료
    if args.list_models:
        list_available_models()
        return
    
    print("=" * 50)
    print("소성가공 압출공정 불량 예측")
    print("=" * 50)
    
    # 사용 가능한 모델 목록 표시
    available_models = list_available_models()
    
    # 1. 모델 로드
    model = load_model(args.model)
    
    # 모델 정보 출력
    model_name = args.model.replace('_model.pkl', '').replace('_', ' ').title()
    print(f"\n사용 모델: {model_name}")
    if args.threshold != 0.5:
        print(f"불량 판정 임계값: {args.threshold}")
    
    # 2. 데이터 로드
    X, dates, true_labels, has_label = load_data(args.data)
    
    # 3. 예측 수행
    predictions, probs = predict(model, X, threshold=args.threshold)
    
    # 4. 결과 저장
    result_df = save_predictions(predictions, probs, dates, true_labels, args.output)
    
    # 5. 결과 요약 출력
    print_prediction_summary(predictions, true_labels, probs)
    
    # 6. 고위험 샘플 상세 정보 출력
    print_high_risk_samples(result_df, top_n=args.top_n, threshold=args.high_risk_threshold)
    
    # 7. 간단 경고 메시지
    if probs is not None:
        high_risk_count = np.sum(probs >= args.high_risk_threshold)
        if high_risk_count > 0:
            print(f"\n⚠️  총 {high_risk_count}개의 고위험 샘플이 발견되었습니다!")
            print(f"   위 상세 정보 및 'high_risk_samples.csv' 파일을 확인하세요.")
    
    print("\n" + "=" * 50)
    print("예측 완료!")
    print("=" * 50)
    print(f"\n📄 예측 결과 파일: {args.output}")
    
    # 추천 모델 안내
    if args.model == 'random_forest_model.pkl':
        print("\n💡 Tip: 더 높은 성능을 위해 XGBoost 또는 LightGBM 모델을 사용해보세요!")
        print("   예: python predict_defect.py --model xgboost_model.pkl")


if __name__ == "__main__":
    main()
