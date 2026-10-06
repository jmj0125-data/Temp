import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울 연평균기온을 이용해 선형회귀 모델을 만들고 "
    "과거 데이터를 학습한 모델이 최근 기온을 얼마나 잘 예측하는지 평가합니다."
)


# =========================================================
# 데이터 주소
# =========================================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# =========================================================
# 데이터 불러오기
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write(e)
    st.stop()


# =========================================================
# 2025년까지의 데이터만 사용
# =========================================================
df = df[
    df["연도"] <= 2025
].copy()


# =========================================================
# 연도별 평균기온 + 관측일수
# =========================================================
yearly = (
    df.groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)


# =========================================================
# 관측일수 300일 미만인 연도 제외
# =========================================================
yearly = yearly[
    yearly["관측일수"] >= 300
].copy()

yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


# =========================================================
# 분석에 필요한 기간 확인
# =========================================================
train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()

train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()

test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()


# =========================================================
# 선형회귀 함수
# 독립변수 = 1908년부터 지난 연수
# =========================================================
def make_regression(data):
    x = (
        data["연도"].to_numpy() - 1908
    )

    y = data["평균기온"].to_numpy()

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    return slope, intercept


# =========================================================
# 50년 모델
# =========================================================
slope_50, intercept_50 = make_regression(
    train_50
)


# =========================================================
# 100년 모델
# =========================================================
slope_100, intercept_100 = make_regression(
    train_100
)


# =========================================================
# 테스트 데이터 예측
# =========================================================
test_x = (
    test["연도"].to_numpy() - 1908
)

test_y = test["평균기온"].to_numpy()


pred_50 = (
    slope_50 * test_x
    + intercept_50
)

pred_100 = (
    slope_100 * test_x
    + intercept_100
)


# =========================================================
# 평가 지표 계산
# =========================================================
mae_50 = mean_absolute_error(
    test_y,
    pred_50
)

mse_50 = mean_squared_error(
    test_y,
    pred_50
)

r2_50 = r2_score(
    test_y,
    pred_50
)


mae_100 = mean_absolute_error(
    test_y,
    pred_100
)

mse_100 = mean_squared_error(
    test_y,
    pred_100
)

r2_100 = r2_score(
    test_y,
    pred_100
)


# =========================================================
# 제목
# =========================================================
st.header("1. 학습 데이터와 테스트 데이터")


st.markdown(
    """
**훈련 데이터**
- 최근 50년: **1956~2005년**
- 최근 100년: **1906~2005년**

**공통 테스트 데이터**
- **2006~2025년**

두 모델 모두 2006~2025년을 한 번도 학습하지 않고,
오직 1906~2005년 또는 1956~2005년의 자료만 이용하여
테스트 데이터를 예측합니다.
"""
)


# =========================================================
# 데이터 개수
# =========================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "50년 훈련 데이터",
        f"{len(train_50)}개 연도"
    )

with c2:
    st.metric(
        "100년 훈련 데이터",
        f"{len(train_100)}개 연도"
    )

with c3:
    st.metric(
        "공통 테스트 데이터",
        f"{len(test)}개 연도"
    )


# =========================================================
# 기울기 비교
# =========================================================
st.header("2. 회귀선의 기울기 비교")

st.write(
    "기울기를 이해하기 쉽도록 **100년에 기온이 몇 °C 변하는가**로 환산했습니다."
)


slope_50_100 = slope_50 * 100
slope_100_100 = slope_100 * 100


c1, c2 = st.columns(2)

with c1:
    st.metric(
        "최근 50년 학습",
        f"{slope_50_100:+.2f} °C / 100년",
        border=True
    )

with c2:
    st.metric(
        "최근 100년 학습",
        f"{slope_100_100:+.2f} °C / 100년",
        border=True
    )


st.caption(
    "양수는 시간이 지날수록 연평균기온이 높아지는 경향을 의미합니다."
)


# =========================================================
# 회귀선 비교 그래프
# =========================================================
st.subheader("훈련 데이터와 회귀선")


graph_years = np.arange(
    1906,
    2026
)

graph_x = graph_years - 1908


graph_pred_50 = (
    slope_50 * graph_x
    + intercept_50
)

graph_pred_100 = (
    slope_100 * graph_x
    + intercept_100
)


fig_train = go.Figure()


# 실제 데이터
fig_train.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=6),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "평균기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)


# 50년 회귀선
fig_train.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_pred_50,
        mode="lines",
        name="1956~2005 학습 회귀선",
        line=dict(
            width=3
        )
    )
)


# 100년 회귀선
fig_train.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_pred_100,
        mode="lines",
        name="1906~2005 학습 회귀선",
        line=dict(
            width=3,
            dash="dash"
        )
    )
)


# 테스트 구간 시작선
fig_train.add_vline(
    x=2005.5,
    line_dash="dot",
    line_width=2,
    annotation_text="테스트 시작"
)


fig_train.update_layout(
    title="서울 연평균기온과 학습 데이터별 회귀선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    height=600,
    hovermode="x unified"
)


st.plotly_chart(
    fig_train,
    width="stretch"
)


# =========================================================
# 테스트 데이터 예측 그래프
# =========================================================
st.header("3. 테스트 데이터 예측 결과")

st.write(
    "두 회귀선이 학습하지 않은 **2006~2025년 실제 연평균기온**을 "
    "얼마나 잘 따라가는지 비교합니다."
)


fig_test = go.Figure()


# 실제 테스트 데이터
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="lines+markers",
        name="실제 기온",
        line=dict(
            width=3
        ),
        marker=dict(size=7),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "실제 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)


# 50년 모델 예측
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines+markers",
        name="50년 학습 모델 예측",
        line=dict(
            width=2
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "50년 모델: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)


# 100년 모델 예측
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines+markers",
        name="100년 학습 모델 예측",
        line=dict(
            width=2,
            dash="dash"
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "100년 모델: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)


fig_test.update_layout(
    title="2006~2025년 테스트 데이터 예측",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(
        dtick=1
    ),
    height=600,
    hovermode="x unified"
)


st.plotly_chart(
    fig_test,
    width="stretch"
)


# =========================================================
# 성능 평가
# =========================================================
st.header("4. 테스트 데이터 예측 성능 비교")

st.write(
    """
**MAE**: 실제값과 예측값의 차이를 절댓값으로 평균낸 값입니다.  
작을수록 좋습니다.

**MSE**: 예측 오차를 제곱하여 평균낸 값입니다.  
큰 오차에 더 큰 벌점을 주며, 작을수록 좋습니다.

**R²**: 모델이 실제 기온의 변동을 얼마나 설명하는지를 나타냅니다.  
1에 가까울수록 좋습니다.
"""
)


# =========================================================
# 성능 표
# =========================================================
performance = pd.DataFrame(
    {
        "모델": [
            "최근 50년 학습",
            "최근 100년 학습"
        ],
        "학습 기간": [
            "1956~2005",
            "1906~2005"
        ],
        "테스트 기간": [
            "2006~2025",
            "2006~2025"
        ],
        "기울기 (°C/100년)": [
            slope_50_100,
            slope_100_100
        ],
        "MAE (°C)": [
            mae_50,
            mae_100
        ],
        "MSE (°C²)": [
            mse_50,
            mse_100
        ],
        "R²": [
            r2_50,
            r2_100
        ]
    }
)


st.dataframe(
    performance.style.format(
        {
            "기울기 (°C/100년)": "{:+.3f}",
            "MAE (°C)": "{:.3f}",
            "MSE (°C²)": "{:.3f}",
            "R²": "{:.3f}"
        }
    ),
    width="stretch",
    hide_index=True
)


# =========================================================
# 지표를 나란히 크게 표시
# =========================================================
st.subheader("모델별 성능")


left, right = st.columns(2)


with left:
    st.markdown("### 최근 50년 학습")
    
    a, b, c = st.columns(3)

    with a:
        st.metric(
            "MAE",
            f"{mae_50:.2f} °C"
        )

    with b:
        st.metric(
            "MSE",
            f"{mse_50:.2f}"
        )

    with c:
        st.metric(
            "R²",
            f"{r2_50:.3f}"
        )


with right:
    st.markdown("### 최근 100년 학습")
    
    a, b, c = st.columns(3)

    with a:
        st.metric(
            "MAE",
            f"{mae_100:.2f} °C"
        )

    with b:
        st.metric(
            "MSE",
            f"{mse_100:.2f}"
        )

    with c:
        st.metric(
            "R²",
            f"{r2_100:.3f}"
        )


# =========================================================
# 어느 모델이 더 좋은지 자동 판단
# =========================================================
st.header("5. 두 모델의 비교")


if mae_50 < mae_100:
    mae_result = "최근 50년 모델"
elif mae_100 < mae_50:
    mae_result = "최근 100년 모델"
else:
    mae_result = "두 모델이 동일"


if mse_50 < mse_100:
    mse_result = "최근 50년 모델"
elif mse_100 < mse_50:
    mse_result = "최근 100년 모델"
else:
    mse_result = "두 모델이 동일"


if r2_50 > r2_100:
    r2_result = "최근 50년 모델"
elif r2_100 > r2_50:
    r2_result = "최근 100년 모델"
else:
    r2_result = "두 모델이 동일"


comparison = pd.DataFrame(
    {
        "평가 기준": [
            "MAE",
            "MSE",
            "R²"
        ],
        "더 좋은 모델": [
            mae_result,
            mse_result,
            r2_result
        ],
        "판단 기준": [
            "작을수록 좋음",
            "작을수록 좋음",
            "클수록 좋음"
        ]
    }
)


st.table(comparison)


# =========================================================
# 해석
# =========================================================
st.subheader("💡 결과 해석")

st.markdown(
    f"""
- **최근 50년 모델의 기울기**는 100년에 **{slope_50_100:+.2f}°C**입니다.
- **최근 100년 모델의 기울기**는 100년에 **{slope_100_100:+.2f}°C**입니다.
- 2006~2025년을 테스트 데이터로 사용했을 때,
  **MAE가 더 작은 모델은 {mae_result}**입니다.
- **MSE가 더 작은 모델은 {mse_result}**입니다.
- **R²가 더 큰 모델은 {r2_result}**입니다.

즉, 단순히 오래된 데이터를 많이 사용하는 것이 항상 더 좋은 것은 아니며,
어느 기간의 데이터를 학습에 사용하는 것이 최근 기온을 예측하는 데
더 적합한지는 실제 테스트 성능을 비교해서 판단할 수 있습니다.
"""
)


# =========================================================
# 데이터 처리 기준
# =========================================================
with st.expander("📋 데이터 처리 기준"):
    st.write("• 2025년까지의 데이터만 사용")
    st.write("• 연간 관측일수가 300일 미만인 연도는 제외")
    st.write("• 일별 평균기온을 연도별로 평균하여 연평균기온 계산")
    st.write("• 50년 모델: 1956~2005년 학습")
    st.write("• 100년 모델: 1906~2005년 학습")
    st.write("• 두 모델의 공통 테스트: 2006~2025년")
    st.write("• 독립변수: 1908년부터 지난 연수")
    st.write("• 모델: 선형회귀")
