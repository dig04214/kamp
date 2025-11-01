"""
PassorFail 값에 따른 변수들의 추이 시각화 스크립트
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def create_output_directory(dir_name='trend_graphs'):
    """그래프 저장을 위한 디렉토리 생성"""
    output_dir = Path(dir_name)
    output_dir.mkdir(exist_ok=True)
    return output_dir

def load_data(file_path):
    """데이터 로드"""
    print(f"데이터 로딩 중: {file_path}")
    df = pd.read_csv(file_path)
    print(f"데이터 형태: {df.shape}")
    print(f"PassorFail 값 분포:\n{df['passorfail'].value_counts()}")
    return df

def plot_variable_trends_by_class(df, output_dir):
    """PassorFail 값에 따른 각 변수의 추이 시각화"""
    # PassorFail 값별로 데이터 분리
    pass_data = df[df['passorfail'] == 1]
    fail_data = df[df['passorfail'] == 0]
    
    # 숫자형 컬럼만 선택 (date와 passorfail 제외)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('passorfail')
    
    print(f"\n시각화할 변수 개수: {len(numeric_cols)}")
    
    # 각 변수에 대한 시계열 플롯
    for col in numeric_cols:
        plt.figure(figsize=(15, 6))
        
        # Pass(1) 데이터 플롯
        plt.subplot(1, 2, 1)
        plt.plot(pass_data.index[:], pass_data[col].iloc[:], 
                alpha=0.7, linewidth=0.5, color='green')
        plt.title(f'{col} - Pass (정상)', fontsize=12, fontweight='bold')
        plt.xlabel('샘플 인덱스')
        plt.ylabel(col)
        plt.grid(True, alpha=0.3)
        
        # Fail(0) 데이터 플롯
        plt.subplot(1, 2, 2)
        plt.plot(fail_data.index[:], fail_data[col].iloc[:], 
                alpha=0.7, linewidth=0.5, color='red')
        plt.title(f'{col} - Fail (불량)', fontsize=12, fontweight='bold')
        plt.xlabel('샘플 인덱스')
        plt.ylabel(col)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 파일명에서 특수문자 제거
        safe_col_name = col.replace('.', '_').replace('/', '_')
        plt.savefig(output_dir / f'trend_{safe_col_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ {col} 그래프 저장 완료")

def plot_variable_distribution_comparison(df, output_dir):
    """PassorFail 값에 따른 변수 분포 비교"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('passorfail')
    
    print("\n분포 비교 그래프 생성 중...")
    
    for col in numeric_cols:
        plt.figure(figsize=(12, 5))
        
        # 히스토그램 비교
        plt.subplot(1, 2, 1)
        plt.hist(df[df['passorfail'] == 1][col], bins=50, alpha=0.6, 
                label='Pass (정상)', color='green', density=True)
        plt.hist(df[df['passorfail'] == 0][col], bins=50, alpha=0.6, 
                label='Fail (불량)', color='red', density=True)
        plt.xlabel(col)
        plt.ylabel('밀도')
        plt.title(f'{col} - 분포 비교', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 박스플롯 비교
        plt.subplot(1, 2, 2)
        data_to_plot = [df[df['passorfail'] == 1][col], 
                       df[df['passorfail'] == 0][col]]
        bp = plt.boxplot(data_to_plot, patch_artist=True)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][1].set_facecolor('red')
        plt.xticks([1, 2], ['Pass (정상)', 'Fail (불량)'])
        plt.ylabel(col)
        plt.title(f'{col} - 박스플롯 비교', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        safe_col_name = col.replace('.', '_').replace('/', '_')
        plt.savefig(output_dir / f'distribution_{safe_col_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ {col} 분포 그래프 저장 완료")

def plot_overall_comparison(df, output_dir):
    """전체 변수들의 평균값 비교"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('passorfail')
    
    # Pass와 Fail의 평균값 계산
    pass_means = df[df['passorfail'] == 1][numeric_cols].mean()
    fail_means = df[df['passorfail'] == 0][numeric_cols].mean()
    
    # 차이가 큰 순서로 정렬
    mean_diff = abs(pass_means - fail_means)
    sorted_cols = mean_diff.sort_values(ascending=False).index
    
    # 상위 20개 변수만 표시
    top_n = min(20, len(sorted_cols))
    top_cols = sorted_cols[:top_n]
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(top_cols))
    width = 0.35
    
    plt.bar(x - width/2, pass_means[top_cols], width, label='Pass (정상)', 
           color='green', alpha=0.7)
    plt.bar(x + width/2, fail_means[top_cols], width, label='Fail (불량)', 
           color='red', alpha=0.7)
    
    plt.xlabel('변수')
    plt.ylabel('평균값')
    plt.title('PassorFail 값에 따른 변수별 평균값 비교 (차이가 큰 상위 20개)', 
             fontsize=14, fontweight='bold')
    plt.xticks(x, top_cols, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'overall_mean_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("\n✓ 전체 평균값 비교 그래프 저장 완료")

def plot_correlation_heatmap(df, output_dir):
    """PassorFail별 상관관계 히트맵"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Pass 데이터 상관관계
    pass_corr = df[df['passorfail'] == 1][numeric_cols].corr()
    sns.heatmap(pass_corr, ax=axes[0], cmap='coolwarm', center=0, 
               square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    axes[0].set_title('상관관계 히트맵 - Pass (정상)', fontsize=14, fontweight='bold')
    
    # Fail 데이터 상관관계
    fail_corr = df[df['passorfail'] == 0][numeric_cols].corr()
    sns.heatmap(fail_corr, ax=axes[1], cmap='coolwarm', center=0, 
               square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    axes[1].set_title('상관관계 히트맵 - Fail (불량)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("✓ 상관관계 히트맵 저장 완료")

def main():
    """메인 실행 함수"""
    print("="*60)
    print("PassorFail 값에 따른 변수 추이 시각화 시작")
    print("="*60)
    
    # 데이터 로드
    df = load_data('소성가공 압출공정 데이터셋.csv')
    
    # 출력 디렉토리 생성
    output_dir = create_output_directory('trend_graphs')
    print(f"\n그래프 저장 디렉토리: {output_dir.absolute()}")
    
    # 1. 전체 평균값 비교
    print("\n" + "="*60)
    print("1. 전체 평균값 비교 그래프 생성")
    print("="*60)
    plot_overall_comparison(df, output_dir)
    
    # 2. 상관관계 히트맵
    print("\n" + "="*60)
    print("2. 상관관계 히트맵 생성")
    print("="*60)
    plot_correlation_heatmap(df, output_dir)
    
    # 3. 변수별 추이 그래프
    print("\n" + "="*60)
    print("3. 변수별 시계열 추이 그래프 생성")
    print("="*60)
    plot_variable_trends_by_class(df, output_dir)
    
    # 4. 변수별 분포 비교 그래프
    print("\n" + "="*60)
    print("4. 변수별 분포 비교 그래프 생성")
    print("="*60)
    plot_variable_distribution_comparison(df, output_dir)
    
    print("\n" + "="*60)
    print("모든 그래프 생성 완료!")
    print(f"저장 위치: {output_dir.absolute()}")
    print("="*60)

if __name__ == "__main__":
    main()
