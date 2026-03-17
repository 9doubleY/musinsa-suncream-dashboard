import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────
# 페이지 설정
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="무신사 남성 선케어 EDA 대시보드",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────
# 색상·상수
# ────────────────────────────────────────────────────────────
COLORS = {
    "A_top50":            "#E74C3C",
    "B_oliveyoung_cross": "#3498DB",
    "C_potential":        "#2ECC71",
}
LAYER_KR = {
    "A_top50":            "A: 무신사 TOP50",
    "B_oliveyoung_cross": "B: 올영 교차",
    "C_potential":        "C: 잠재군",
}
LAYER_ORDER = ["A_top50", "B_oliveyoung_cross", "C_potential"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ────────────────────────────────────────────────────────────
# CSS — 다크 테마 + 커스텀 카드
# ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 전체 배경 */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background: #1a1d27;
}
/* 상단 헤더 */
.main-header {
    background: linear-gradient(135deg, #1a1d27 0%, #2c3e50 100%);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    border-left: 6px solid #E74C3C;
}
.main-header h1 {
    margin: 0; font-size: 2rem; font-weight: 800;
    background: linear-gradient(90deg, #E74C3C, #F39C12);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.main-header p { margin: 6px 0 0; color: #95a5a6; font-size: 0.95rem; }
/* KPI 카드 */
.kpi-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 18px 20px;
    border-top: 3px solid;
    text-align: center;
    height: 100%;
}
.kpi-label { font-size: 0.75rem; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }
.kpi-value { font-size: 2rem; font-weight: 800; margin: 6px 0 2px; }
.kpi-delta { font-size: 0.8rem; }
/* 섹션 제목 */
.section-title {
    font-size: 1.15rem; font-weight: 700; color: #ecf0f1;
    border-left: 4px solid; padding-left: 10px;
    margin: 20px 0 12px;
}
/* 인사이트 박스 */
.insight-box {
    background: #1e2130;
    border-left: 4px solid #F39C12;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: #bdc3c7;
    line-height: 1.6;
}
.insight-box strong { color: #F39C12; }
/* 탭 */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
/* 사이드바 타이틀 */
.sidebar-section {
    font-size: 0.75rem; font-weight: 700;
    color: #7f8c8d; text-transform: uppercase;
    letter-spacing: 1.5px; margin: 16px 0 6px;
}
/* 테이블 */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
/* 스크롤 */
.stPlotlyChart { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# 데이터 로드 (캐시)
# ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    prod      = pd.read_csv(f"{DATA_DIR}/product_agg.csv")
    monthly   = pd.read_csv(f"{DATA_DIR}/monthly_reviews.csv")
    survey    = pd.read_csv(f"{DATA_DIR}/survey_agg.csv")
    shap_df   = pd.read_csv(f"{DATA_DIR}/shap_importance.csv")
    rf_df     = pd.read_csv(f"{DATA_DIR}/rf_importance.csv")
    c_rank    = pd.read_csv(f"{DATA_DIR}/c_potential_ranking.csv")
    price_d   = pd.read_csv(f"{DATA_DIR}/price_dist.csv")
    skin_l    = pd.read_csv(f"{DATA_DIR}/skin_layer.csv")
    cluster   = pd.read_csv(f"{DATA_DIR}/cluster_stats.csv")
    price_pos = pd.read_csv(f"{DATA_DIR}/price_position.csv")
    with open(f"{DATA_DIR}/keywords.json", encoding="utf-8") as f:
        kw = json.load(f)
    return prod, monthly, survey, shap_df, rf_df, c_rank, price_d, skin_l, cluster, price_pos, kw

prod, monthly, survey, shap_df, rf_df, c_rank, price_d, skin_l, cluster, price_pos, kw = load_data()

# ────────────────────────────────────────────────────────────
# 헬퍼
# ────────────────────────────────────────────────────────────
def layer_color_map():
    return {LAYER_KR[k]: v for k, v in COLORS.items()}

def add_layer_kr(df, col="layer"):
    df = df.copy()
    df[col+"_kr"] = df[col].map(LAYER_KR)
    return df

def plotly_layout(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#ecf0f1")),
        paper_bgcolor="#1e2130",
        plot_bgcolor="#1e2130",
        font=dict(color="#bdc3c7", size=11),
        height=height,
        margin=dict(l=40, r=30, t=50 if title else 20, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#2c3e50", zerolinecolor="#2c3e50")
    fig.update_yaxes(gridcolor="#2c3e50", zerolinecolor="#2c3e50")
    return fig

# ────────────────────────────────────────────────────────────
# 사이드바
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☀️ 무신사 선케어")
    st.markdown("**EDA 분석 대시보드**")
    st.divider()

    st.markdown('<div class="sidebar-section">📌 레이어 필터</div>', unsafe_allow_html=True)
    sel_layers = st.multiselect(
        "분석 레이어 선택",
        options=LAYER_ORDER,
        default=LAYER_ORDER,
        format_func=lambda x: LAYER_KR[x],
    )

    st.markdown('<div class="sidebar-section">💰 가격대 필터</div>', unsafe_allow_html=True)
    price_range = st.slider("가격 범위 (원)", 5000, 65000, (5000, 65000), step=1000)

    st.markdown('<div class="sidebar-section">⭐ 최소 리뷰 수</div>', unsafe_allow_html=True)
    min_reviews = st.slider("최소 리뷰 수", 0, 500, 0, step=10)

    st.divider()
    st.markdown('<div class="sidebar-section">📊 기준일</div>', unsafe_allow_html=True)
    st.caption("2026년 03월 14일")

    st.markdown('<div class="sidebar-section">🔢 데이터 현황</div>', unsafe_allow_html=True)
    st.caption(f"총 리뷰: **26,916건**")
    st.caption(f"분석 제품: **168개**")
    st.caption(f"TOP50 리뷰: **19,279건**")

# 필터 적용
prod_f = prod[
    (prod["layer"].isin(sel_layers)) &
    (prod["price"] >= price_range[0]) &
    (prod["price"] <= price_range[1]) &
    (prod["review_count"] >= min_reviews)
].copy()
prod_f = add_layer_kr(prod_f)
price_pos_f = price_pos[price_pos["layer"].isin(sel_layers)]

# ────────────────────────────────────────────────────────────
# 헤더
# ────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>☀️ 무신사 남성 선케어 EDA 대시보드</h1>
    <p>하위 제품 판매순위 상승 전략 도출 · 시장 구조 분석 · 소비자 니즈 분석 · RF+SHAP 모델 해석</p>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# KPI 카드 행
# ────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)

total_products = len(prod_f)
avg_price = prod_f["price"].mean() if len(prod_f) else 0
avg_grade = prod_f["grade_mean"].mean() if len(prod_f) else 0
avg_proba = prod_f["top_proba_mean"].mean() if len(prod_f) else 0
total_reviews = prod_f["review_count"].sum()
avg_photo = prod_f["photo_rate"].mean() * 100 if len(prod_f) else 0

kpis = [
    (k1, "분석 제품 수",      f"{total_products}개",         "#E74C3C",  "필터 적용 기준"),
    (k2, "평균 가격",          f"₩{avg_price:,.0f}",          "#3498DB",  "가격 필터 적용"),
    (k3, "평균 평점",          f"{avg_grade:.3f} / 5.0",      "#2ECC71",  "별점 1~5"),
    (k4, "TOP50 예측확률",    f"{avg_proba:.3f}",             "#F39C12",  "RF 모델 결과"),
    (k5, "총 리뷰 수",         f"{int(total_reviews):,}건",   "#9B59B6",  "필터 적용 기준"),
    (k6, "사진 리뷰율",        f"{avg_photo:.1f}%",           "#1ABC9C",  "전체 리뷰 기준"),
]
for col, label, val, color, sub in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color:{color}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color}">{val}</div>
            <div class="kpi-delta" style="color:#7f8c8d">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# 탭 구성
# ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 시장 구조",
    "🎯 제품 포지셔닝",
    "💬 소비자 리뷰 분석",
    "🤖 ML 모델 해석",
    "🚀 C군 전략 대시보드",
    "📋 제품 데이터 탐색",
])

# ════════════════════════════════════════════════════════════
# TAB 1: 시장 구조
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title" style="border-color:#E74C3C">💰 가격 분포 분석</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # 박스플롯
        df_box = add_layer_kr(prod_f)
        fig = px.box(
            df_box, x="layer_kr", y="price",
            color="layer_kr",
            color_discrete_map=layer_color_map(),
            points="outliers",
            labels={"layer_kr": "레이어", "price": "가격 (원)"},
            category_orders={"layer_kr": [LAYER_KR[l] for l in LAYER_ORDER if l in sel_layers]},
        )
        fig = plotly_layout(fig, "레이어별 가격 분포 (박스플롯)")
        fig.update_traces(marker=dict(opacity=0.6))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # 가격대 Stacked Bar
        price_d_filt = price_d.copy()
        cols_sel = ["price_band"] + [c for c in LAYER_ORDER if c in sel_layers and c in price_d.columns]
        price_d_filt = price_d_filt[cols_sel]

        fig2 = go.Figure()
        for layer in LAYER_ORDER:
            if layer in sel_layers and layer in price_d_filt.columns:
                fig2.add_trace(go.Bar(
                    x=price_d_filt["price_band"],
                    y=price_d_filt[layer],
                    name=LAYER_KR[layer],
                    marker_color=COLORS[layer],
                    opacity=0.85,
                ))
        fig2.update_layout(barmode="stack")
        fig2 = plotly_layout(fig2, "가격대별 제품 분포")
        st.plotly_chart(fig2, use_container_width=True)

    # 가격 히스토그램 (레이어 오버레이)
    fig3 = go.Figure()
    for layer in LAYER_ORDER:
        if layer in sel_layers:
            d = prod_f[prod_f["layer"] == layer]["price"]
            fig3.add_trace(go.Histogram(
                x=d, name=LAYER_KR[layer],
                marker_color=COLORS[layer], opacity=0.6,
                nbinsx=20, xbins=dict(start=5000, end=70000, size=3000),
            ))
    fig3.update_layout(barmode="overlay")
    fig3 = plotly_layout(fig3, "레이어별 가격 분포 오버레이 히스토그램", height=320)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-title" style="border-color:#3498DB">📉 할인율 & 리뷰 볼륨 분석</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        # 할인율 바이올린
        fig4 = go.Figure()
        for layer in LAYER_ORDER:
            if layer in sel_layers:
                d = prod_f[prod_f["layer"] == layer]["saleRate"].dropna()
                fig4.add_trace(go.Violin(
                    y=d, name=LAYER_KR[layer],
                    box_visible=True, meanline_visible=True,
                    fillcolor=COLORS[layer], opacity=0.7,
                    line_color=COLORS[layer],
                ))
        fig4 = plotly_layout(fig4, "레이어별 할인율 분포 (%)")
        st.plotly_chart(fig4, use_container_width=True)

    with c4:
        # 월별 리뷰 추이
        monthly_filt = monthly[monthly["layer"].isin(sel_layers)].copy()
        monthly_filt["layer_kr"] = monthly_filt["layer"].map(LAYER_KR)
        monthly_pivot = monthly_filt.pivot_table(index="yearmonth", columns="layer_kr", values="count", fill_value=0)
        monthly_pivot = monthly_pivot.sort_index()

        fig5 = go.Figure()
        for layer in LAYER_ORDER:
            if layer in sel_layers:
                lkr = LAYER_KR[layer]
                if lkr in monthly_pivot.columns:
                    fig5.add_trace(go.Scatter(
                        x=monthly_pivot.index,
                        y=monthly_pivot[lkr],
                        name=lkr,
                        mode="lines+markers",
                        line=dict(color=COLORS[layer], width=2),
                        marker=dict(size=4),
                        fill="tozeroy",
                        fillcolor="rgba({},{},{},0.12)".format(
                            int(COLORS[layer][1:3], 16),
                            int(COLORS[layer][3:5], 16),
                            int(COLORS[layer][5:7], 16),
                        ),
                    ))
        fig5 = plotly_layout(fig5, "월별 리뷰 발생 추이")
        fig5.update_xaxes(tickangle=-45)
        st.plotly_chart(fig5, use_container_width=True)

    # 인사이트
    st.markdown("""
    <div class="insight-box">
    💡 <strong>핵심 인사이트</strong><br>
    • A_top50 중앙가격 <strong>20,650원</strong> vs C_potential 중앙가격 <strong>19,050원</strong> — 가격 차이는 1,600원에 불과 (가격이 순위 결정 요인 아님)<br>
    • A_top50 평균 할인율 <strong>29.1%</strong>로 C군(27.6%)보다 높음 — 무신사 쿠폰/기획 전략이 상위 진입에 기여<br>
    • C_potential의 상세 리뷰 비중이 높은 것은 열성 팬의 존재를 의미하나, 볼륨 부족으로 노출 알고리즘에서 불리
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 2: 제품 포지셔닝
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title" style="border-color:#F39C12">🗺️ 가격-리뷰 포지셔닝 매트릭스</div>', unsafe_allow_html=True)

    prod_f2 = prod_f[prod_f["review_count"] > 0].copy()

    fig_pos = px.scatter(
        prod_f2,
        x="price", y="grade_mean",
        size="review_count",
        color="layer_kr",
        color_discrete_map=layer_color_map(),
        hover_name="goodsName",
        hover_data={"price": ":,.0f", "grade_mean": ":.3f",
                    "review_count": True, "layer_kr": False,
                    "top_proba_mean": ":.3f"},
        labels={"price": "가격 (원)", "grade_mean": "평균 평점",
                "review_count": "리뷰 수", "layer_kr": "레이어"},
        size_max=50,
        opacity=0.75,
    )
    med_price = prod_f2["price"].median()
    med_grade = prod_f2["grade_mean"].median()
    fig_pos.add_vline(x=med_price, line_dash="dash", line_color="#7f8c8d", opacity=0.5)
    fig_pos.add_hline(y=med_grade, line_dash="dash", line_color="#7f8c8d", opacity=0.5)

    # 사분면 레이블
    for txt, ax, ay, ac in [
        ("고가·고평점", med_price * 1.12, med_grade + 0.008, "#F39C12"),
        ("저가·고평점", med_price * 0.35,  med_grade + 0.008, "#2ECC71"),
        ("고가·저평점", med_price * 1.12, med_grade - 0.015, "#E74C3C"),
        ("저가·저평점", med_price * 0.35,  med_grade - 0.015, "#95a5a6"),
    ]:
        fig_pos.add_annotation(x=ax, y=ay, text=txt,
                               showarrow=False, font=dict(color=ac, size=10))

    fig_pos = plotly_layout(fig_pos, "가격 × 평점 포지셔닝 매트릭스  (버블 크기 = 리뷰 수)", height=520)
    st.plotly_chart(fig_pos, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title" style="border-color:#9B59B6">📊 판매순위 vs 가격</div>', unsafe_allow_html=True)
        fig_rank = px.scatter(
            prod_f2, x="collect_rank", y="price",
            color="layer_kr",
            color_discrete_map=layer_color_map(),
            hover_name="goodsName",
            hover_data={"collect_rank": True, "price": ":,.0f", "layer_kr": False},
            labels={"collect_rank": "판매 순위", "price": "가격 (원)", "layer_kr": "레이어"},
            opacity=0.7, size_max=8,
        )
        fig_rank = plotly_layout(fig_rank, "")
        st.plotly_chart(fig_rank, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title" style="border-color:#1ABC9C">📦 제품 유형별 구성</div>', unsafe_allow_html=True)
        prod_f2_type = prod_f2.copy()
        prod_f2_type["product_type"] = prod_f2_type["goodsName"].apply(lambda x:
            "선스틱" if "선스틱" in str(x) or ("스틱" in str(x) and "쿠션" not in str(x)) else
            "선쿠션" if "쿠션" in str(x) else
            "선앰플/세럼" if "앰플" in str(x) or "세럼" in str(x) else
            "세트/기획" if any(k in str(x) for k in ["SET","세트","2개","2pack","기획"]) else
            "선크림"
        )
        type_layer = prod_f2_type.groupby(["product_type","layer_kr"]).size().reset_index(name="count")
        fig_type = px.bar(
            type_layer, x="product_type", y="count", color="layer_kr",
            color_discrete_map=layer_color_map(),
            barmode="stack",
            labels={"product_type":"제품 유형","count":"제품 수","layer_kr":"레이어"},
        )
        fig_type = plotly_layout(fig_type, "")
        st.plotly_chart(fig_type, use_container_width=True)

    # 레이어별 핵심 지표 레이더
    st.markdown('<div class="section-title" style="border-color:#E74C3C">🕸️ 레이어별 핵심 지표 레이더</div>', unsafe_allow_html=True)

    metrics = ["grade_mean","photo_rate","first_review_rate","spread_score","sticky_score","irritate_score"]
    metric_labels = ["평균평점","사진리뷰율","첫구매리뷰율","발림성","무끈적임","무자극"]
    # 0~1 정규화
    radar_data = prod_f.groupby("layer")[metrics].mean()
    for m in metrics:
        mn, mx = radar_data[m].min(), radar_data[m].max()
        if mx > mn:
            radar_data[m] = (radar_data[m] - mn) / (mx - mn)

    fig_radar = go.Figure()
    for layer in LAYER_ORDER:
        if layer in sel_layers and layer in radar_data.index:
            vals = radar_data.loc[layer, metrics].tolist()
            vals_closed = vals + [vals[0]]
            labels_closed = metric_labels + [metric_labels[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed, theta=labels_closed,
                fill="toself", name=LAYER_KR[layer],
                line_color=COLORS[layer], fillcolor=COLORS[layer],
                opacity=0.35,
            ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#1e2130",
            radialaxis=dict(visible=True, range=[0,1], color="#7f8c8d",
                            gridcolor="#2c3e50", tickfont=dict(size=9)),
            angularaxis=dict(color="#bdc3c7"),
        ),
        paper_bgcolor="#1e2130",
        font=dict(color="#bdc3c7"),
        height=420,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=60, t=30, b=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <strong>포지셔닝 인사이트</strong><br>
    • C군의 무끈적임·무자극 점수가 A군을 상회 — <strong>제품 품질 자체는 충분</strong>하나 노출·리뷰 볼륨에서 열위<br>
    • 판매순위와 가격의 상관관계는 약함 (상관계수 ~0.1) — 가격 인하보다 <strong>마케팅·리뷰 전략</strong>이 더 효과적<br>
    • 세트/기획 제품이 A군의 20%를 차지 — 기획 구성이 순위 견인에 유효한 전략
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 3: 소비자 리뷰 분석
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title" style="border-color:#2ECC71">🔑 레이어별 TF-IDF 상위 키워드</div>', unsafe_allow_html=True)

    kw_col1, kw_col2, kw_col3 = st.columns(3)
    for col, layer in zip([kw_col1, kw_col2, kw_col3], LAYER_ORDER):
        if layer in sel_layers and layer in kw:
            with col:
                kw_df = pd.DataFrame(kw[layer]).head(15)
                fig_kw = px.bar(
                    kw_df[::-1], x="score", y="keyword",
                    orientation="h",
                    color="score",
                    color_continuous_scale=["#1e3a5f","#2980B9","#85C1E9"] if layer == "B_oliveyoung_cross"
                        else ["#1a4a2e","#2ECC71","#ABEBC6"] if layer == "C_potential"
                        else ["#641e16","#E74C3C","#F1948A"],
                    labels={"score":"TF-IDF 점수","keyword":"키워드"},
                )
                fig_kw = plotly_layout(fig_kw, f"{LAYER_KR[layer]} — 상위 키워드", height=380)
                fig_kw.update_coloraxes(showscale=False)
                st.plotly_chart(fig_kw, use_container_width=True)

    st.markdown('<div class="section-title" style="border-color:#9B59B6">🗂️ K-Means 클러스터 분석 (TF-IDF + Centroid)</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        # 클러스터별 레이어 비율 히트맵
        cluster_f = cluster.copy()
        cluster_f["label"] = cluster_f["cluster"].astype(str) + ": " + cluster_f["cluster_name"]
        available_layers = [l for l in LAYER_ORDER if l in cluster_f.columns and l in sel_layers]
        cluster_pct = cluster_f[available_layers].div(cluster_f[available_layers].sum(axis=1), axis=0) * 100
        cluster_pct["label"] = cluster_f["label"]

        z_vals = cluster_pct[available_layers].values
        x_labels = [LAYER_KR[l] for l in available_layers]
        y_labels = cluster_pct["label"].tolist()

        fig_hm = go.Figure(go.Heatmap(
            z=z_vals, x=x_labels, y=y_labels,
            colorscale="Blues",
            text=np.round(z_vals, 1),
            texttemplate="%{text}%",
            showscale=True,
        ))
        fig_hm = plotly_layout(fig_hm, "클러스터 × 레이어 비율 히트맵 (%)", height=360)
        st.plotly_chart(fig_hm, use_container_width=True)

    with c2:
        # 클러스터별 평균 평점
        fig_cl = px.bar(
            cluster_f, x="grade", y="label",
            orientation="h",
            color="grade",
            color_continuous_scale=["#641e16","#E74C3C","#F39C12","#2ECC71"],
            range_color=[4.7, 5.0],
            labels={"grade":"평균 평점","label":"클러스터"},
        )
        fig_cl = plotly_layout(fig_cl, "클러스터별 평균 평점", height=360)
        fig_cl.update_coloraxes(showscale=False)
        fig_cl.update_xaxes(range=[4.7, 5.02])
        st.plotly_chart(fig_cl, use_container_width=True)

    st.markdown('<div class="section-title" style="border-color:#1ABC9C">🧴 피부타입 & 설문 분석</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        # 피부타입 히트맵
        skin_filt = skin_l.copy()
        skin_cols = [l for l in LAYER_ORDER if l in skin_filt.columns and l in sel_layers]
        skin_pct = skin_filt[skin_cols].div(skin_filt[skin_cols].sum(axis=1), axis=0) * 100

        fig_skin = go.Figure(go.Heatmap(
            z=skin_pct.values,
            x=[LAYER_KR[c] for c in skin_cols],
            y=skin_filt["skinType"].tolist(),
            colorscale="YlOrRd",
            text=np.round(skin_pct.values, 1),
            texttemplate="%{text}%",
            showscale=True,
        ))
        fig_skin = plotly_layout(fig_skin, "피부타입 × 레이어 비율 히트맵 (%)", height=320)
        st.plotly_chart(fig_skin, use_container_width=True)

    with c4:
        # 설문 레이더
        survey_filt = survey[survey["layer"].isin(sel_layers)].copy()
        survey_metrics = ["발림성_score","끈적임_score","자극_score"]
        survey_labels  = ["발림성","무끈적임","무자극"]

        fig_sv = go.Figure()
        for _, row in survey_filt.iterrows():
            layer = row["layer"]
            vals  = [row[m] for m in survey_metrics]
            vals_c = vals + [vals[0]]
            lbls_c = survey_labels + [survey_labels[0]]
            fig_sv.add_trace(go.Scatterpolar(
                r=vals_c, theta=lbls_c,
                fill="toself", name=LAYER_KR[layer],
                line_color=COLORS[layer], fillcolor=COLORS[layer],
                opacity=0.4,
            ))
        fig_sv.update_layout(
            polar=dict(
                bgcolor="#1e2130",
                radialaxis=dict(visible=True, range=[3, 5], color="#7f8c8d",
                                gridcolor="#2c3e50", tickfont=dict(size=9)),
                angularaxis=dict(color="#bdc3c7"),
            ),
            paper_bgcolor="#1e2130",
            font=dict(color="#bdc3c7"),
            height=320,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=50, r=50, t=30, b=30),
        )
        st.plotly_chart(fig_sv, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <strong>소비자 니즈 인사이트</strong><br>
    • 클러스터 C4(상세 리뷰)에서 C군 비중 <strong>50.5%</strong> — 소수 열성 팬이 긴 리뷰 작성<br>
    • A군 리뷰에는 <strong>"촉촉"·"발림성"</strong> 등 감각적 언어가 고빈도 — 제품 설명에 이 언어 반영 권장<br>
    • 설문: C군의 무자극 평점 <strong>4.86</strong>점 (A군 4.64점보다 높음) — 제품 품질 우위를 마케팅에 적극 활용해야 함
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 4: ML 모델 해석
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title" style="border-color:#9B59B6">🤖 Pseudo-label Random Forest 개요</div>', unsafe_allow_html=True)

    info_c1, info_c2, info_c3, info_c4 = st.columns(4)
    model_kpis = [
        ("CV ROC-AUC", "0.6765", "5-Fold 평균", "#9B59B6"),
        ("AUC 표준편차", "±0.028", "안정적 성능", "#3498DB"),
        ("A군 예측확률", "0.562", "평균값", "#E74C3C"),
        ("C군 예측확률", "0.418", "평균값", "#2ECC71"),
    ]
    for col, (lbl, val, sub, clr) in zip([info_c1,info_c2,info_c3,info_c4], model_kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color:{clr}">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value" style="color:{clr}">{val}</div>
                <div class="kpi-delta" style="color:#7f8c8d">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title" style="border-color:#F39C12">🌲 RF 특성 중요도 (Gini)</div>', unsafe_allow_html=True)
        rf_top = rf_df.head(20).copy()
        rf_top["feature_clean"] = rf_top["feature"].str.replace("tfidf_","TF: ")
        fig_rf = px.bar(
            rf_top[::-1], x="importance", y="feature_clean",
            orientation="h",
            color="importance",
            color_continuous_scale=["#1a4a2e","#2ECC71","#F39C12","#E74C3C"],
            labels={"importance":"Feature Importance","feature_clean":"특성명"},
        )
        fig_rf = plotly_layout(fig_rf, "RF 특성 중요도 TOP 20", height=520)
        fig_rf.update_coloraxes(showscale=False)
        st.plotly_chart(fig_rf, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title" style="border-color:#E74C3C">🔍 SHAP 특성 중요도</div>', unsafe_allow_html=True)
        shap_top = shap_df.head(20).copy()
        shap_top["feature_clean"] = shap_top["feature"].str.replace("tfidf_","TF: ")
        fig_shap = px.bar(
            shap_top[::-1], x="importance", y="feature_clean",
            orientation="h",
            color="importance",
            color_continuous_scale=["#1a3a5c","#3498DB","#85C1E9","#F0F3FF"],
            labels={"importance":"Mean |SHAP value|","feature_clean":"특성명"},
        )
        fig_shap = plotly_layout(fig_shap, "SHAP 특성 중요도 TOP 20", height=520)
        fig_shap.update_coloraxes(showscale=False)
        st.plotly_chart(fig_shap, use_container_width=True)

    # SHAP vs RF 비교 테이블
    st.markdown('<div class="section-title" style="border-color:#1ABC9C">📊 RF vs SHAP 특성 중요도 비교</div>', unsafe_allow_html=True)

    top_n = st.slider("표시할 특성 수", 5, 20, 10)
    rf_top_n  = rf_df.head(top_n).copy().reset_index(drop=True)
    shap_top_n = shap_df.head(top_n).copy().reset_index(drop=True)
    rf_top_n["rank"] = range(1, top_n+1)
    shap_top_n["rank"] = range(1, top_n+1)

    compare_df = pd.DataFrame({
        "순위": range(1, top_n+1),
        "RF 특성": rf_top_n["feature"].str.replace("tfidf_","TF: "),
        "RF 중요도": rf_top_n["importance"].round(4),
        "SHAP 특성": shap_top_n["feature"].str.replace("tfidf_","TF: "),
        "SHAP 중요도": shap_top_n["importance"].round(4),
    })

    fig_compare = make_subplots(rows=1, cols=2, subplot_titles=["RF Feature Importance","SHAP Feature Importance"])
    fig_compare.add_trace(
        go.Bar(x=rf_top_n["importance"][:top_n], y=rf_top_n["feature"].str.replace("tfidf_","TF: ")[:top_n],
               orientation="h", marker_color="#2ECC71", opacity=0.8, name="RF"),
        row=1, col=1,
    )
    fig_compare.add_trace(
        go.Bar(x=shap_top_n["importance"][:top_n], y=shap_top_n["feature"].str.replace("tfidf_","TF: ")[:top_n],
               orientation="h", marker_color="#3498DB", opacity=0.8, name="SHAP"),
        row=1, col=2,
    )
    fig_compare = plotly_layout(fig_compare, "", height=420)
    fig_compare.update_layout(showlegend=False)
    st.plotly_chart(fig_compare, use_container_width=True)

    # ANOVA 결과 테이블
    st.markdown('<div class="section-title" style="border-color:#E74C3C">📐 ANOVA 검정 결과</div>', unsafe_allow_html=True)
    anova_data = {
        "변수": ["평점(Grade)", "텍스트 길이", "좋아요 수", "TOP50 예측확률"],
        "A_top50 평균": [4.857, 57.4, 0.055, 0.562],
        "B_올영 평균":  [4.892, 53.0, 0.038, 0.501],
        "C_잠재 평균":  [4.875, 127.9, 0.110, 0.418],
        "F 통계량":    [9.338, 1357.8, 52.99, 3787.7],
        "p 값":        ["0.0001", "<0.0001", "<0.0001", "<0.0001"],
        "유의성":      ["***", "***", "***", "***"],
    }
    anova_df = pd.DataFrame(anova_data)
    st.dataframe(
        anova_df.style
            .format({"A_top50 평균":":.3f","B_올영 평균":":.3f","C_잠재 평균":":.3f","F 통계량":":.1f"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("""
    <div class="insight-box">
    💡 <strong>모델 해석 인사이트</strong><br>
    • SHAP 1위: <strong>is_first_review</strong> — 첫 구매 리뷰가 TOP50 리뷰를 가장 잘 구별함 → 신규 고객 유입이 핵심<br>
    • SHAP 2위: <strong>발림성_score</strong> — 발림성 만족도가 높을수록 TOP50 제품 리뷰와 유사<br>
    • SHAP 5위: <strong>TF: "촉촉"</strong> — 감각적 키워드가 TOP50 리뷰를 특징짓는 핵심 언어<br>
    • ANOVA 텍스트 길이 F=<strong>1,357.8</strong> — C군 리뷰가 훨씬 길지만 TOP50 진입에는 연결되지 않음 (볼륨 부족이 원인)
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 5: C군 전략 대시보드
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title" style="border-color:#2ECC71">🎯 C군 TOP50 진입 가능성 순위 (RF 예측확률 기반)</div>', unsafe_allow_html=True)

    top_n_c = st.slider("표시할 제품 수", 5, 30, 15, key="c_slider")
    c_top = c_rank.head(top_n_c).copy()
    c_top["short_name"] = c_top["goodsName"].apply(lambda x: x[:20]+"…" if len(str(x))>20 else x)

    fig_c = go.Figure()
    fig_c.add_trace(go.Bar(
        y=c_top["short_name"][::-1],
        x=c_top["top_proba_mean"][::-1],
        orientation="h",
        marker=dict(
            color=c_top["top_proba_mean"][::-1],
            colorscale=[[0,"#1a4a2e"],[0.5,"#2ECC71"],[1.0,"#F39C12"]],
            showscale=True,
            colorbar=dict(title="예측확률", tickfont=dict(color="#bdc3c7"), title_font=dict(color="#bdc3c7")),
        ),
        text=[f"{v:.3f}" for v in c_top["top_proba_mean"][::-1]],
        textposition="outside",
        textfont=dict(color="#bdc3c7", size=10),
    ))
    fig_c.add_vline(x=0.530, line_dash="dash", line_color="#F39C12",
                   annotation_text="진입 임계 (0.530)", annotation_font_color="#F39C12")
    fig_c = plotly_layout(fig_c, f"C군 TOP50 진입 가능성 TOP {top_n_c}", height=max(400, top_n_c * 28))
    st.plotly_chart(fig_c, use_container_width=True)

    # 전략 매트릭스
    st.markdown('<div class="section-title" style="border-color:#F39C12">🗺️ 전략 포지셔닝 매트릭스</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        fig_strat = px.scatter(
            prod_f,
            x="collect_rank", y="top_proba_mean",
            color="layer_kr",
            color_discrete_map=layer_color_map(),
            size="review_count",
            size_max=30,
            hover_name="goodsName",
            hover_data={"collect_rank":True,"top_proba_mean":":.3f",
                        "price":":,.0f","review_count":True,"layer_kr":False},
            labels={"collect_rank":"현재 판매순위","top_proba_mean":"TOP50 예측확률","layer_kr":"레이어"},
            opacity=0.75,
        )
        q75 = prod_f["top_proba_mean"].quantile(0.75)
        fig_strat.add_hline(y=q75, line_dash="dot", line_color="#F39C12",
                            annotation_text=f"75th 분위수 ({q75:.3f})",
                            annotation_font_color="#F39C12")
        fig_strat = plotly_layout(fig_strat, "판매순위 × TOP50 예측확률 (버블=리뷰수)", height=480)
        st.plotly_chart(fig_strat, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title" style="border-color:#E74C3C">📌 단기 전략 액션 플랜</div>', unsafe_allow_html=True)
        actions = [
            ("🥇", "리뷰 캠페인 집중 투자", "구매 후 7일 내 리뷰 시 포인트 지급\n사진 포함 시 2배 지급", "#E74C3C"),
            ("📸", "사진 리뷰 유도", "QR코드로 '사진 리뷰 이벤트' 안내\n실사용 인증샷 목표 35% 이상", "#F39C12"),
            ("🏷️", "신규 쿠폰 발급", "10~15% 쿠폰으로 첫구매 유도\nis_first_review 30% 달성", "#3498DB"),
            ("✍️", "상품 설명 최적화", "'촉촉·보송·발림성' 키워드 반영\n소비자 리뷰 언어 패턴 유도", "#2ECC71"),
            ("📦", "스타터 세트 출시", "미니 샘플 + 본품 기획\n진입 장벽 낮춰 신규 구매 확대", "#9B59B6"),
        ]
        for icon, title, desc, clr in actions:
            st.markdown(f"""
            <div style="background:#1e2130;border-left:3px solid {clr};
                        border-radius:6px;padding:10px 14px;margin:8px 0">
                <div style="font-size:0.9rem;font-weight:700;color:{clr}">{icon} {title}</div>
                <div style="font-size:0.78rem;color:#95a5a6;margin-top:3px;
                            white-space:pre-line">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # C군 상세 지표 비교
    st.markdown('<div class="section-title" style="border-color:#1ABC9C">📊 C군 상위 10개 제품 상세 지표</div>', unsafe_allow_html=True)
    c_detail = c_rank.head(10)[["goodsName","price","collect_rank","top_proba_mean",
                                  "grade_mean","review_count","photo_rate",
                                  "spread_score","sticky_score","irritate_score"]].copy()
    c_detail.columns = ["제품명","가격","현재순위","TOP50예측","평균평점",
                         "리뷰수","사진율","발림성","무끈적임","무자극"]
    c_detail["제품명"] = c_detail["제품명"].apply(lambda x: x[:22]+"…" if len(str(x))>22 else x)
    st.dataframe(
        c_detail.style
            .format({
                "가격": "{:,.0f}원",
                "TOP50예측": "{:.3f}",
                "평균평점": "{:.3f}",
                "사진율": "{:.1%}",
                "발림성": "{:.2f}",
                "무끈적임": "{:.2f}",
                "무자극": "{:.2f}",
            }),
        use_container_width=True, hide_index=True,
    )

    # KPI 목표
    st.markdown('<div class="section-title" style="border-color:#9B59B6">🎯 KPI 목표 트래커</div>', unsafe_allow_html=True)
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
    kpi_items = [
        ("단기 리뷰 목표", 68, 200, "건", "#E74C3C"),
        ("사진 리뷰 목표", 15, 35, "%", "#F39C12"),
        ("순위 목표 (단기)", 350, 150, "위", "#3498DB"),
        ("TOP50 진입 목표", 0, 2, "개", "#2ECC71"),
    ]
    for col, (title, cur, goal, unit, clr) in zip([kpi_c1,kpi_c2,kpi_c3,kpi_c4], kpi_items):
        with col:
            pct = min(int(cur/goal*100), 100)
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color:{clr}">
                <div class="kpi-label">{title}</div>
                <div class="kpi-value" style="color:{clr}">{cur}{unit}</div>
                <div style="background:#2c3e50;border-radius:4px;height:6px;margin:6px 0">
                    <div style="background:{clr};width:{pct}%;height:6px;border-radius:4px"></div>
                </div>
                <div class="kpi-delta" style="color:#7f8c8d">목표: {goal}{unit} ({pct}% 달성)</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    💡 <strong>C군 전략 우선순위</strong><br>
    • <strong>1순위: 마다가스카르 센텔라 실키핏 선스틱</strong> — 예측확률 0.551, 주류가격 17,000원, 선스틱 제형(검증된 카테고리)<br>
    • <strong>2순위: 파데프리 선크림 2팩</strong> — 평점 4.977(최고수준), 예측확률 0.536, 리뷰 45건(집중 투자시 빠른 성과 기대)<br>
    • <strong>3순위: 인텐시브 롱래스팅 선스크린 EX (단품+2팩 시너지)</strong> — 두 SKU 연계 전략으로 교차 리뷰 유도
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 6: 제품 데이터 탐색
# ════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title" style="border-color:#3498DB">🔍 인터랙티브 제품 탐색기</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        search_term = st.text_input("🔍 제품명 검색", placeholder="예: 선스틱, 토코보, 선크림…")
    with col_f2:
        sort_by = st.selectbox("정렬 기준", ["collect_rank","top_proba_mean","grade_mean","review_count","price"])
    with col_f3:
        sort_order = st.radio("정렬 방향", ["오름차순","내림차순"], horizontal=True)

    show_df = prod_f.copy()
    if search_term:
        show_df = show_df[show_df["goodsName"].str.contains(search_term, na=False, case=False) |
                          show_df["brandName"].str.contains(search_term, na=False, case=False)]

    show_df = show_df.sort_values(sort_by, ascending=(sort_order=="오름차순"))

    display_cols = ["goodsName","brandName","layer","collect_rank","price","saleRate",
                    "reviewScore","review_count","grade_mean","top_proba_mean",
                    "photo_rate","spread_score"]
    display_df = show_df[display_cols].copy()
    display_df.columns = ["제품명","브랜드","레이어","판매순위","가격","할인율",
                           "리뷰점수","리뷰수","평균평점","TOP50확률","사진율","발림성"]
    display_df["레이어"] = display_df["레이어"].map(LAYER_KR)
    display_df["제품명"] = display_df["제품명"].apply(lambda x: x[:28]+"…" if len(str(x))>28 else x)

    st.markdown(f"**총 {len(display_df)}개 제품** (필터·검색 적용)", unsafe_allow_html=True)
    st.dataframe(
        display_df.style
            .format({
                "가격": "{:,.0f}원",
                "할인율": "{:.0f}%",
                "평균평점": "{:.3f}",
                "TOP50확률": "{:.3f}",
                "사진율": "{:.1%}",
                "발림성": "{:.2f}",
            }),
        use_container_width=True,
        height=480,
    )

    # 산점도 — 사용자 커스텀
    st.markdown('<div class="section-title" style="border-color:#F39C12">📈 커스텀 산점도</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    numeric_cols = ["price","saleRate","review_count","grade_mean","top_proba_mean",
                    "photo_rate","text_len_mean","spread_score","sticky_score","collect_rank"]
    with sc1:
        x_axis = st.selectbox("X축", numeric_cols, index=0)
    with sc2:
        y_axis = st.selectbox("Y축", numeric_cols, index=4)
    with sc3:
        size_by = st.selectbox("버블 크기", ["review_count","top_proba_mean","grade_mean"], index=0)

    fig_custom = px.scatter(
        prod_f, x=x_axis, y=y_axis,
        size=size_by, color="layer_kr",
        color_discrete_map=layer_color_map(),
        hover_name="goodsName",
        opacity=0.7, size_max=40,
        labels={"layer_kr":"레이어"},
    )
    fig_custom = plotly_layout(fig_custom, f"{x_axis} vs {y_axis}", height=450)
    st.plotly_chart(fig_custom, use_container_width=True)

    # 브랜드별 분석
    st.markdown('<div class="section-title" style="border-color:#9B59B6">🏷️ 브랜드별 현황</div>', unsafe_allow_html=True)
    brand_agg = prod_f.groupby(["brandName","layer"]).agg(
        제품수=("goodsNo","count"),
        평균가격=("price","mean"),
        평균TOP50확률=("top_proba_mean","mean"),
        평균리뷰수=("review_count","mean"),
    ).reset_index()
    brand_agg["layer_kr"] = brand_agg["layer"].map(LAYER_KR)

    top_brands = brand_agg.groupby("brandName")["제품수"].sum().nlargest(20).index
    brand_plot = brand_agg[brand_agg["brandName"].isin(top_brands)]

    fig_brand = px.bar(
        brand_plot, x="brandName", y="제품수",
        color="layer_kr",
        color_discrete_map=layer_color_map(),
        barmode="stack",
        labels={"brandName":"브랜드","제품수":"제품 수","layer_kr":"레이어"},
    )
    fig_brand.update_xaxes(tickangle=-40)
    fig_brand = plotly_layout(fig_brand, "브랜드별 제품 분포 (TOP 20 브랜드)", height=380)
    st.plotly_chart(fig_brand, use_container_width=True)

# ────────────────────────────────────────────────────────────
# 하단 푸터
# ────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#4a5568;font-size:0.78rem;padding:12px 0">
    ☀️ 무신사 남성 선케어 EDA 대시보드 &nbsp;|&nbsp;
    분석 기준일: 2026.03.14 &nbsp;|&nbsp;
    분석 방법: TF-IDF · K-Means Centroid · Pseudo-label RF · SHAP · ANOVA &nbsp;|&nbsp;
    데이터: 168개 제품 · 26,916건 리뷰
</div>
""", unsafe_allow_html=True)
