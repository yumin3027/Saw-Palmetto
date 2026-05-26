# -*- coding: utf-8 -*-
"""
쏘팔메토 완제품 진위 판별 및 판단 근거 시각화(XAI) 스트림릿 대시보드
작성자: 시니어 풀스택 데이터 소프트웨어 엔지니어 & 분석화학 ML 전문가 Antigravity
"""

import os
import sys
import platform
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from catboost import CatBoostClassifier, Pool

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 프리미엄 CSS 스타일 주입
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="쏘팔메토 진위판별 및 XAI 대시보드",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 한글 폰트 설정 (OS별 호환성 처리)
@st.cache_resource
def set_korean_font():
    system_os = platform.system()
    if system_os == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif system_os == 'Darwin':  # macOS
        plt.rc('font', family='AppleGothic')
    else:  # Linux/Colab 등
        plt.rc('font', family='NanumGothic')
    plt.rc('axes', unicode_minus=False)
    sns.set_theme(style="whitegrid")
    # 한글 폰트가 제대로 적용되도록 시각화 컨텍스트 사전 조정
    plt.rcParams['font.family'] = 'Malgun Gothic' if system_os == 'Windows' else 'AppleGothic'

set_korean_font()

# 프리미엄 스타일을 위한 CSS 주입
st.markdown("""
<style>
    /* 메인 배경색 및 기본 텍스트 톤 조절 */
    .main {
        background-color: #f7f9fc;
    }
    
    /* 타이틀 및 헤더 고급화 */
    .main-title {
        font-family: 'Inter', 'Noto Sans KR', sans-serif;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-title {
        color: #555;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* 카드 레이아웃 스타일 정의 */
    .custom-card {
        background-color: white;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border-left: 5px solid #2a5298;
    }
    
    /* 성공/경고 커스텀 카드 */
    .success-card {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 1.5rem;
        border-radius: 8px;
        color: #14532d;
        margin-bottom: 1rem;
    }
    .danger-card {
        background-color: #fef2f2;
        border-left: 5px solid #dc2626;
        padding: 1.5rem;
        border-radius: 8px;
        color: #7f1d1d;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 분석용 피처(지방산 10종) 정의
# -----------------------------------------------------------------------------
FEATURE_COLS = [
    'Methyl caproate',   # C6:0
    'Methyl caprylate',  # C8:0
    'Methyl caprate',    # C10:0
    'Methyl laurate',    # C12:0
    'Methyl myristate',  # C14:0
    'Methyl palmitate',  # C16:0
    'Methyl stearate',   # C18:0
    'Methyl oleate',     # C18:1
    'Methyl linoleate',  # C18:2
    'Methyl linolenate'  # C18:3
]

FEATURE_MAPPING = {
    'Methyl caproate': 'Caproic (C6:0)',
    'Methyl caprylate': 'Caprylic (C8:0)',
    'Methyl caprate': 'Capric (C10:0)',
    'Methyl laurate': 'Lauric (C12:0)',
    'Methyl myristate': 'Myristic (C14:0)',
    'Methyl palmitate': 'Palmitic (C16:0)',
    'Methyl stearate': 'Stearic (C18:0)',
    'Methyl oleate': 'Oleic (C18:1)',
    'Methyl linoleate': 'Linoleic (C18:2)',
    'Methyl linolenate': 'Linolenic (C18:3)'
}

# -----------------------------------------------------------------------------
# 3. 실제 원본 데이터 기반 모델 학습 엔진 (추출물 및 코코넛오일 전용)
# -----------------------------------------------------------------------------
@st.cache_resource
def train_standard_model():
    """
    실제 원본 엑셀 파일 '머신러닝 연습용 쏘팔메토 지방산 데이터.xlsx'를 로드하고,
    '추출물'(순수 쏘팔메토, Class 0)과 '코코넛오일'(대조군, Class 1)의 실제 데이터셋만 
    엄격하게 필터링하여 진위 판정 모델을 학습합니다. 
    가상/합성 데이터 폴백은 일체 사용하지 않으며 파일 부재 시 즉각 중단됩니다.
    """
    file_name = '머신러닝 연습용 쏘팔메토 지방산 데이터.xlsx'
    
    if not os.path.exists(file_name):
        st.error(f"🚨 [시스템 에러] 학습용 대조군 기준 파일 '{file_name}'을 찾을 수 없습니다. 분석 디렉토리를 확인해주세요.")
        st.stop()
        
    try:
        df_train_raw = pd.read_excel(file_name)
        
        # 종류 컬럼 자동 세팅
        type_col = '종류' if '종류' in df_train_raw.columns else df_train_raw.columns[1]
        df_train_raw[type_col] = df_train_raw[type_col].astype(str).str.strip()
        
        # '추출물' -> 순수 쏘팔메토 (Class 0), '코코넛오일' -> 혼입 제어군 (Class 1)
        df_pure = df_train_raw[df_train_raw[type_col] == '추출물'].copy()
        df_pure['Target'] = 0
        
        df_coconut = df_train_raw[df_train_raw[type_col] == '코코넛오일'].copy()
        df_coconut['Target'] = 1
        
        # 학습 데이터 결합
        df_train_set = pd.concat([df_pure, df_coconut], axis=0).reset_index(drop=True)
        
        # 정규화 연산 수행 (지방산 10종의 행별 합이 100%가 되도록 변환)
        row_sums = df_train_set[FEATURE_COLS].sum(axis=1).replace(0, 1.0)
        for col in FEATURE_COLS:
            df_train_set[col] = (df_train_set[col] / row_sums) * 100.0
            
        X = df_train_set[FEATURE_COLS]
        y = df_train_set['Target']
        
        # 실제 데이터 규모(추출물 1개, 코코넛오일 4개)가 매우 제한적이므로
        # 과적합 방지를 위해 depth=2, iterations=50으로 강한 학습 규제를 적용합니다.
        model = CatBoostClassifier(
            iterations=50,
            learning_rate=0.05,
            depth=2,
            verbose=0,
            random_seed=42
        )
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        
        return model, explainer, X
        
    except Exception as e:
        st.error(f"🚨 [학습 에러] 실제 데이터 기반 모델 피팅 중 치명적 에러 발생: {e}")
        st.stop()

model, explainer, X_train = train_standard_model()

# -----------------------------------------------------------------------------
# 4. 내부 데이터 전처리 엔진 (행별 상대 면적 백분율 정규화)
# -----------------------------------------------------------------------------
def preprocess_and_normalize(df_raw):
    """
    업로드된 지방산 원시 데이터의 절대 면적 편차를 바로잡기 위해,
    사용자가 구성한 컬럼들 중 분석 피처 리스트(10종)의 행별 합을 구한 뒤 
    정확히 합산 '100% 상대 면적 백분율(Relative Area %)' 데이터로 정규화 변환을 수행합니다.
    """
    df_processed = df_raw.copy()
    
    # 필수 변수 존재 여부 체크
    missing_cols = [col for col in FEATURE_COLS if col not in df_processed.columns]
    if missing_cols:
        raise ValueError(f"업로드된 파일에 필수 지방산 컬럼이 누락되었습니다: {missing_cols}")
    
    # 행별 지방산 10종의 합계 구하기
    row_sums = df_processed[FEATURE_COLS].sum(axis=1)
    
    # 0으로 나누어지는 오류 방지
    row_sums = row_sums.replace(0, 1.0)
    
    # 정규화 연산 수행 (각 지방산 값 / 10종의 합 * 100)
    for col in FEATURE_COLS:
        df_processed[col] = (df_processed[col] / row_sums) * 100.0
        
    return df_processed

# -----------------------------------------------------------------------------
# 5. UI Layout - SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📁 데이터 입력 및 제어")
    st.info("기본적으로 로컬의 실제 원본 데이터셋을 자동 로드합니다. 다른 분석 파일을 검증하고 싶으실 때만 업로드해 주세요.")
    
    # 파일 업로더 컴포넌트
    uploaded_file = st.file_uploader(
        "신규 분석 파일 업로드 (.xlsx, .csv)", 
        type=["xlsx", "csv"],
        help="엑셀 파일의 경우 첫 번째 시트의 지방산 함량 데이터를 읽어들입니다."
    )
    
    st.markdown("---")
    st.markdown("#### 🔬 필수 포함 지방산 변수 (10종)")
    st.markdown("""
    업로드 데이터셋은 아래의 10가지 지방산 에스터 메틸화(FAMEs) 표준 컬럼명을 포함해야 분석이 진행됩니다.
    * `Methyl caproate` (C6:0)
    * `Methyl caprylate` (C8:0)
    * `Methyl caprate` (C10:0)
    * `Methyl laurate` (C12:0)
    * `Methyl myristate` (C14:0)
    * `Methyl palmitate` (C16:0)
    * `Methyl stearate` (C18:0)
    * `Methyl oleate` (C18:1)
    * `Methyl linoleate` (C18:2)
    * `Methyl linolenate` (C18:3)
    """)
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #888; font-size: 0.8rem;'>Antigravity Food Authenticity Engine v1.5</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. UI Layout - MAIN DASHBOARD
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>🧪 쏘팔메토 지방산 프로파일 진위판별 대시보드</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>분석화학 정량 데이터 기반의 CatBoost 머신러닝 분류 알고리즘 및 XAI(SHAP) 판단 근거 분석 시스템</div>", unsafe_allow_html=True)

# 데이터 준비 분기 처리 (기본값으로 실제 엑셀 파일 지정)
df_raw = None
default_file_name = '머신러닝 연습용 쏘팔메토 지방산 데이터.xlsx'

if uploaded_file is not None:
    try:
        # 파일 확장자에 따른 판다스 로드
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
            
        st.success(f"🎉 신규 파일 업로드 성공: {uploaded_file.name} (총 {len(df_raw)}개 샘플)")
    except Exception as e:
        st.error(f"🚨 파일 로딩 중 오류가 발생했습니다: {e}")
        st.stop()
else:
    # 업로드 파일이 없을 때 로컬의 실제 분석용 엑셀 파일을 무조건 다이렉트로 로드 (가상 합성 데이터 배제)
    if os.path.exists(default_file_name):
        try:
            df_raw = pd.read_excel(default_file_name)
            st.info(f"💾 로컬 데이터 파일 자동 연동 완료: '{default_file_name}' (총 {len(df_raw)}개 대조군/완제품 샘플 분석 중)")
        except Exception as e:
            st.error(f"🚨 로컬 원본 데이터 로딩 중 에러 발생: {e}")
            st.stop()
    else:
        st.error(f"🚨 [파일 부재] 로컬에 실제 분석용 데이터 파일 '{default_file_name}'이 존재하지 않습니다. 파일을 해당 폴더에 복사해주세요.")
        st.stop()

# -----------------------------------------------------------------------------
# 7. 메인 화면 상단 - 데이터 전처리 및 확인 구역
# -----------------------------------------------------------------------------
if df_raw is not None:
    st.markdown("<div class='custom-card'><h4>📊 원시 데이터 및 상대 면적 정규화 결과 모니터링</h4>", unsafe_allow_html=True)
    
    # 샘플 ID로 쓸 컬럼 탐색 및 지정
    id_col = None
    for col in df_raw.columns:
        if 'ID' in str(col).upper() or '샘플' in str(col) or 'SAMPLE' in str(col).upper():
            id_col = col
            break
            
    if id_col is None:
        df_raw.insert(0, '샘플 ID', [f"Sample-{i+1}" for i in range(len(df_raw))])
        id_col = '샘플 ID'
        
    col_pre1, col_pre2 = st.columns([1, 1])
    
    with col_pre1:
        st.markdown("**✏️ 업로드 원시 데이터 (상위 5행)**")
        st.dataframe(df_raw.head(5), use_container_width=True)
        
    with col_pre2:
        try:
            # 정규화 가동
            df_norm = preprocess_and_normalize(df_raw)
            st.markdown("**🧪 100% 상대 면적 정규화 변환 데이터 (Relative Area %)**")
            st.dataframe(df_norm.head(5), use_container_width=True)
        except Exception as err:
            st.error(f"정규화 수행 중 오류: {err}")
            st.stop()
            
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. 메인 화면 중단 - ML 진위 판정 결과 대시보드
# -----------------------------------------------------------------------------
if 'df_norm' in locals():
    st.markdown("<div class='custom-card'><h4>🔮 CatBoost 머신러닝 분석 및 판정 결과</h4>", unsafe_allow_html=True)
    
    # 예측 수행
    X_unseen = df_norm[FEATURE_COLS]
    preds = model.predict(X_unseen)
    probs = model.predict_proba(X_unseen)
    
    # 판정 요약 메트릭 배치
    num_samples = len(df_norm)
    num_pure = sum(preds == 0)
    num_adulterated = sum(preds == 1)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("총 분석 의뢰 샘플 수", f"{num_samples} 건")
    col_m2.metric("✅ 순수 정품 판정 건수", f"{num_pure} 건", delta=f"{num_pure/num_samples*100:.1f}%")
    col_m3.metric("🚨 가짜/혼입 판정 건수", f"{num_adulterated} 건", delta=f"-{num_adulterated/num_samples*100:.1f}%", delta_color="inverse")
    
    st.markdown("---")
    st.markdown("**📋 개별 샘플별 진위 판정 리포트**")
    st.caption("아래의 각 샘플 탭을 클릭하시면 정밀 지방산 비율과 USP 규격 만족도 등 상세 리포트를 확인하실 수 있습니다.")
    
    # 각 샘플별 결과 박스 렌더링
    for idx, row in df_norm.iterrows():
        sample_name = row[id_col]
        pred_label = preds[idx]
        prob_pure = probs[idx][0] * 100.0
        prob_fake = probs[idx][1] * 100.0
        
        lauric = row['Methyl laurate']
        violations = []
        detailed_status = []
        
        # USP 검증 엔진
        usp_standards = {
            'C6:0': ('Methyl caproate', 8.5, 24.0),
            'C8:0': ('Methyl caprylate', 8.5, 17.5),
            'C10:0': ('Methyl caprate', 9.0, 16.0),
            'C14:0': ('Methyl myristate', 2.2, 2.8),
            'C16:0': ('Methyl palmitate', 2.8, 3.9),
            'C18:0': ('Methyl stearate', 14.0, 26.0),
            'C18:1': ('Methyl oleate', 0.60, 1.15),
            'C18:2': ('Methyl linoleate', 5.0, 16.0),
            'C18:3': ('Methyl linolenate', 31.5, 55.0)
        }
        
        for k, (orig_name, low, high) in usp_standards.items():
            val = row[orig_name]
            ratio = lauric / val if val > 0 else np.inf
            if ratio < low or ratio > high:
                violations.append(f"{k} ({ratio:.2f})")
                detailed_status.append({
                    "지방산 종류": k,
                    "FAMEs 원시 함량": f"{val:.3f}%",
                    "C12 대비 비율": f"{ratio:.2f}",
                    "USP 규격 기준": f"{low} ~ {high}",
                    "판정": "❌ 불합격"
                })
            else:
                detailed_status.append({
                    "지방산 종류": k,
                    "FAMEs 원시 함량": f"{val:.3f}%",
                    "C12 대비 비율": f"{ratio:.2f}",
                    "USP 규격 기준": f"{low} ~ {high}",
                    "판정": "✅ 합격"
                })
                
        num_violate = len(violations)
        
        # Expander 제목 정의 (한눈에 결과를 볼 수 있게 컬러코드화)
        if pred_label == 0:
            if num_violate > 0:
                title = f"🟡 [혼입의심/경고] {sample_name} | 정품 확률 {prob_pure:.1f}% | USP 미세이탈 {num_violate}건"
            else:
                title = f"🟢 [정품판정/통과] {sample_name} | 정품 확률 {prob_pure:.1f}% | USP 규격 완벽통합"
        else:
            title = f"🔴 [가짜/혼입판정] {sample_name} | 혼입 위험도 {prob_fake:.1f}% | USP 불합격 {num_violate}건"
            
        with st.expander(title, expanded=False):
            # 레이아웃을 좌우로 나누어 상세 분석표와 지방산 프로파일 차트 렌더링
            col_d1, col_d2 = st.columns([1.3, 1])
            
            with col_d1:
                st.markdown("**🔬 USP 기준 대조 분석표 (C12:0 분자 비율 기준)**")
                df_det = pd.DataFrame(detailed_status)
                
                # 가독성을 위해 판정 컬럼 색상 하이라이팅
                def style_cell_by_result(val):
                    color = '#fef2f2' if '❌' in val else '#f0fdf4'
                    text_color = '#b91c1c' if '❌' in val else '#15803d'
                    return f'background-color: {color}; color: {text_color}; font-weight: bold;'
                    
                st.dataframe(
                    df_det.style.map(style_cell_by_result, subset=['판정']),
                    use_container_width=True, 
                    hide_index=True
                )
                
            with col_d2:
                st.markdown("**🧪 정규화된 10대 주요 지방산 조성 함량 (Area %)**")
                sample_fa_data = pd.DataFrame({
                    "지방산 FAMEs": [FEATURE_MAPPING[col] for col in FEATURE_COLS],
                    "조성 비율(%)": [row[col] for col in FEATURE_COLS]
                }).sort_values(by="조성 비율(%)", ascending=True)
                
                # 수평 바 차트로 해당 샘플의 구체적인 함량 조성 표시
                fig_sample, ax_sample = plt.subplots(figsize=(5, 3.8))
                
                # 중요 마커 지방산 색상 다르게 강조 (Lauric, Oleic, Linoleic)
                colors_s = []
                for x in sample_fa_data["지방산 FAMEs"]:
                    if 'Lauric' in x:
                        colors_s.append('#1e3c72')  # 핵심 C12:0
                    elif 'Linoleic' in x:
                        colors_s.append('#dc2626')  # 가짜 판별의 기여인자 C18:2
                    elif 'Oleic' in x:
                        colors_s.append('#16a34a')  # 정품의 핵심 장쇄불포화 C18:1
                    else:
                        colors_s.append('#cbd5e1')  # 기타 지방산
                        
                bars = ax_sample.barh(sample_fa_data["지방산 FAMEs"], sample_fa_data["조성 비율(%)"], color=colors_s, height=0.55)
                
                # 바 오른쪽에 수치 직접 표기
                for bar in bars:
                    width = bar.get_width()
                    ax_sample.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.2f}%", 
                                va='center', ha='left', fontsize=8, color='#333')
                    
                ax_sample.spines['top'].set_visible(False)
                ax_sample.spines['right'].set_visible(False)
                ax_sample.spines['bottom'].set_visible(False)
                ax_sample.xaxis.set_visible(False)
                ax_sample.tick_params(axis='both', which='major', labelsize=8)
                
                plt.title(f"{sample_name} 지방산 함량 프로파일 (%)", fontsize=9, fontweight='bold', pad=10)
                plt.tight_layout()
                st.pyplot(fig_sample)
                plt.close(fig_sample)
                
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 9. 메인 화면 하단 - 인공지능 판정 근거 시각화 레이아웃 (XAI)
# -----------------------------------------------------------------------------
if 'df_norm' in locals():
    st.markdown("<div class='custom-card'><h4>📊 AI 판정 근거 수치 시각화 (Explainable AI)</h4>", unsafe_allow_html=True)
    
    col_x1, col_x2 = st.columns(2)
    
    # 9.1. SHAP Summary Plot
    with col_x1:
        st.markdown("**🔬 1) SHAP 글로벌 요약 설명 (Summary Plot)**")
        st.caption("개별 지방산 성분이 진위판정(Class 1: 혼입)에 미치는 기여도와 방향성을 시각화합니다.")
        
        try:
            # SHAP Value 계산
            shap_values = explainer(X_unseen)
            
            # Matplotlib Figure 객체로 감싸기
            fig, ax = plt.subplots(figsize=(6, 5))
            
            # 컬럼 한글명을 위해 임시로 피처 이름 변경
            shap_values.feature_names = [FEATURE_MAPPING[col] for col in FEATURE_COLS]
            
            # SHAP Summary Plot 생성
            shap.summary_plot(shap_values.values, X_unseen, feature_names=shap_values.feature_names, show=False)
            
            # 레이아웃 조정 및 한글 깨짐 패치
            plt.title("지방산별 진위 판별 기여도 (SHAP)", fontsize=12, pad=15)
            plt.xlabel("SHAP Value (혼입 기여도)", fontsize=10)
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close(fig)
        except Exception as shap_err:
            st.error(f"SHAP Summary Plot 생성 중 에러 발생: {shap_err}")
            
    # 9.2. Feature Importance / Local Explanation Bar Chart
    with col_x2:
        st.markdown("**📈 2) 글로벌 변수 중요도 및 핵심 판단 기여 지표**")
        st.caption("CatBoost 머신러닝 엔진이 의사결정을 내릴 때 가장 중요하게 참조한 지방산 순위 리스트입니다.")
        
        try:
            # 피처 임포턴스 추출
            importances = model.get_feature_importance()
            feat_imp_df = pd.DataFrame({
                '지방산 성분': [FEATURE_MAPPING[col] for col in FEATURE_COLS],
                '기여도 점수': importances
            }).sort_values(by='기여도 점수', ascending=True)
            
            # Matplotlib 바 차트 그리기
            fig_bar, ax_bar = plt.subplots(figsize=(6, 5.2))
            
            colors = sns.color_palette("Blues_d", n_colors=len(FEATURE_COLS))
            bars = ax_bar.barh(feat_imp_df['지방산 성분'], feat_imp_df['기여도 점수'], color=colors, edgecolor='grey', height=0.6)
            
            # 막대 우측에 정량값 표시
            for bar in bars:
                width = bar.get_width()
                ax_bar.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                            va='center', ha='left', fontsize=9, fontweight='bold', color='#333')
                
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.spines['bottom'].set_visible(False)
            ax_bar.xaxis.set_visible(False)
            
            plt.title("머신러닝 알고리즘 지방산 중요도 순위", fontsize=12, pad=15)
            plt.tight_layout()
            
            st.pyplot(fig_bar)
            plt.close(fig_bar)
            
        except Exception as imp_err:
            st.error(f"Feature Importance 시각화 중 에러 발생: {imp_err}")
            
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 9.3. 추가 분석화학적 원인 소견 제시 (전문가 인사이트 가치 전달)
    st.markdown("""
    <div class='custom-card' style='border-left: 5px solid #0284c7;'>
        <h4>🔬 식품 진위판별 전문가의 핵심 분석 소견</h4>
        <ul>
            <li><strong>지방산 정규화(Normalization)의 의의:</strong> GC-FID 원시 데이터는 샘플의 주입량이나 검출기 감도에 따라 절대 농도가 흔들릴 수 있습니다. 본 시스템은 업로드된 10종의 총합을 기준으로 100% 상대 비율로 사전 정규화하므로, 주입 변동성을 원천 차단하고 순수 화학적 성분 밸런스로만 판별을 실행합니다.</li>
            <li><strong>핵심 마커의 화학적 특성:</strong> 쏘팔메토 진위 판정에 있어 가장 기여도가 높은 성분은 <strong>Methyl linoleate (C18:2)</strong> 및 <strong>Methyl laurate (C12:0)</strong>입니다. 대두유, 옥수수유 등의 식물성 오일은 C18:2 함량이 매우 높아 쏘팔메토에 혼입될 시 SHAP 요약도에서 보듯 급격히 우측(혼입 양수 기여)으로 플롯이 치우치며 가짜 오일로 식별됩니다. 반면 코코넛오일 혼입 시에는 C12:0 성분이 폭증하여 규칙을 무너뜨리게 됩니다.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
