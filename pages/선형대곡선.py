
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="기온 예측기 - 다항회귀 평가",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기 - 직선과 곡선 비교")

st.write(
    "2005년 이전의 연평균기온만으로 1차, 3차, 9차 회귀모델을 학습하고, "
    "학습에 사용하지 않은 2005년 이후의 데이터로 성능을 평가합니다."
)


# =========================================================
# 데이터
# =========================================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

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
    st.error("기온 데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.stop()


# =========================================================
# 2025년까지 사용
# =========================================================
df = df[
    df["연도"] <= 2025
].copy()


# =========================================================
# 연도별 연평균기온 계산
# 관측일수 300일 미만인 해 제외
# =========================================================
yearly = (
    df.groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

yearly = yearly[
    yearly["관측일수"] >= 300
].copy()

yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


# =========================================================
# 훈련 / 테스트 분리
#
# 훈련: 2005년 이전
# 테스트: 2005년부터
# =========================================================
train = yearly[
    yearly["연도"] < 2005
].copy()

test = yearly[
    yearly["연도"] >= 2005
].copy()


# =========================================================
# 수치 안정화를 위한 연도 변환
#
# 실제 연도 대신
#
# x = (연도 - 2005) / 100
#
# 을 사용한다.
#
# 따라서 2005년은 x=0
# 1905년은 x=-1
# 2105년은 x=1
#
# 9차식에서도 숫자가 지나치게 커지는 것을 방지한다.
# =========================================================
def scaled_year(year):
    return (np.asarray(year) - 2005) / 100.0


x_train = scaled_year(
    train["연도"]
)

y_train = train[
    "평균기온"
].to_numpy()


x_test = scaled_year(
    test["연도"]
)

y_test = test[
    "평균기온"
].to_numpy()


# =========================================================
# 다항회귀 함수
# =========================================================
def fit_polynomial(degree):
    coefficients = np.polyfit(
        x_train,
        y_train,
        degree
    )

    return coefficients


def predict(coefficients, x):
    return np.polyval(
        coefficients,
        x
    )


# =========================================================
# 1차 / 3차 / 9차 모델 학습
# =========================================================
coef_1 = fit_polynomial(1)
coef_3 = fit_polynomial(3)
coef_9 = fit_polynomial(9)


# =========================================================
# 테스트 데이터 예측
# =========================================================
pred_1 = predict(
    coef_1,
    x_test
)

pred_3 = predict(
    coef_3,
    x_test
)

pred_9 = predict(
    coef_9,
    x_test
)


# =========================================================
# 평균 오차 계산
#
# "평균 몇 도나 빗나가는가?"
# = 평균 절대 오차 MAE
# =========================================================
def mae(actual, predicted):
    return np.mean(
        np.abs(actual - predicted)
    )


mae_1 = mae(
    y_test,
    pred_1
)

mae_3 = mae(
    y_test,
    pred_3
)

mae_9 = mae(
    y_test,
    pred_9
)


# =========================================================
# 2050년 예측
# =========================================================
x_2050 = scaled_year(
    2050
)

prediction_2050_1 = predict(
    coef_1,
    [x_2050]
)[0]

prediction_2050_3 = predict(
    coef_3,
    [x_2050]
)[0]

prediction_2050_9 = predict(
    coef_9,
    [x_2050]
)[0]


# =========================================================
# 데이터 개수
# =========================================================
st.header("1. 훈련 데이터와 테스트 데이터")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "훈련 데이터",
        f"{len(train)}개 연도"
    )

    st.write(
        f"기간: **{train['연도'].min()}~"
        f"{train['연도'].max()}년**"
    )

with col2:
    st.metric(
        "테스트 데이터",
        f"{len(test)}개 연도"
    )

    st.write(
        f"기간: **{test['연도'].min()}~"
        f"{test['연도'].max()}년**"
    )


st.info(
    "테스트 데이터는 모델을 만들 때 전혀 사용하지 않았습니다. "
    "모델을 모두 2005년 이전 데이터로 학습한 뒤, "
    "2005년 이후 데이터를 이용해서 채점했습니다."
)


# =========================================================
# 핵심 결과 표
# =========================================================
st.header("2. 1차·3차·9차 회귀 결과")

result = pd.DataFrame(
    {
        "모델": [
            "1차 (직선)",
            "3차 곡선",
            "9차 곡선"
        ],
        "학습 연도 수": [
            len(train),
            len(train),
            len(train)
        ],
        "테스트 평균 절대 오차 (°C)": [
            mae_1,
            mae_3,
            mae_9
        ],
        "2050년 예측값 (°C)": [
            prediction_2050_1,
            prediction_2050_3,
            prediction_2050_9
        ]
    }
)


st.dataframe(
    result.style.format(
        {
            "테스트 평균 절대 오차 (°C)": "{:.3f}",
            "2050년 예측값 (°C)": "{:.2f}"
        }
    ),
    width="stretch",
    hide_index=True
)


# =========================================================
# 큰 숫자로 비교
# =========================================================
st.subheader("📊 모델별 테스트 성능")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### 1차 · 직선")

    st.metric(
        "평균 오차",
        f"{mae_1:.2f} °C"
    )

    st.metric(
        "2050년 예상",
        f"{prediction_2050_1:.2f} °C"
    )

with c2:
    st.markdown("### 3차 · 곡선")

    st.metric(
        "평균 오차",
        f"{mae_3:.2f} °C"
    )

    st.metric(
        "2050년 예상",
        f"{prediction_2050_3:.2f} °C"
    )

with c3:
    st.markdown("### 9차 · 곡선")

    st.metric(
        "평균 오차",
        f"{mae_9:.2f} °C"
    )

    st.metric(
        "2050년 예상",
        f"{prediction_2050_9:.2f} °C"
    )


# =========================================================
# 그래프용 데이터
# =========================================================
graph_years = np.arange(
    yearly["연도"].min(),
    2051
)

graph_x = scaled_year(
    graph_years
)


graph_pred_1 = predict(
    coef_1,
    graph_x
)

graph_pred_3 = predict(
    coef_3,
    graph_x
)

graph_pred_9 = predict(
    coef_9,
    graph_x
)


# =========================================================
# 전체 데이터 + 세 회귀선
# =========================================================
st.header("3. 실제 데이터와 세 회귀선")

fig = go.Figure()


# 실제 훈련 데이터
fig.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["평균기온"],
        mode="markers",
        name="훈련 데이터",
        marker=dict(size=6),
        hovertemplate=(
            "%{x}년<br>"
            "연평균기온: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)


# 실제 테스트 데이터
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="markers",
        name="테스트 데이터",
        marker=dict(size=7),
        hovertemplate=(
            "%{x}년<br>"
            "실제 연평균기온: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)


# 1차
fig.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_pred_1,
        mode="lines",
        name="1차 회귀",
        line=dict(width=3),
        hovertemplate=(
            "%{x}년<br>"
            "1차 예측: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)


# 3차
fig.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_pred_3,
        mode="lines",
        name="3차 회귀",
        line=dict(
            width=3,
            dash="dash"
        ),
        hovertemplate=(
            "%{x}년<br>"
            "3차 예측: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)


# 9차
fig.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_pred_9,
        mode="lines",
        name="9차 회귀",
        line=dict(
            width=3,
            dash="dot"
        ),
        hovertemplate=(
            "%{x}년<br>"
            "9차 예측: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)


# 훈련/테스트 경계
fig.add_vline(
    x=2005,
    line_dash="dash",
    line_width=2,
    annotation_text="2005년: 테스트 시작"
)


fig.update_layout(
    title="서울 연평균기온과 1차·3차·9차 회귀선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    height=650,
    hovermode="x unified"
)


st.plotly_chart(
    fig,
    width="stretch"
)


# =========================================================
# 테스트 구간만 확대
# =========================================================
st.header("4. 테스트 데이터에서 실제값과 예측값 비교")

fig_test = go.Figure()


fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="lines+markers",
        name="실제 기온",
        line=dict(width=4),
        marker=dict(size=7),
        hovertemplate=(
            "%{x}년<br>"
            "실제: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)


fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_1,
        mode="lines",
        name="1차 예측",
        line=dict(width=2),
        hovertemplate=(
            "%{x}년<br>"
            "1차: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)


fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_3,
        mode="lines",
        name="3차 예측",
        line=dict(
            width=2,
            dash="dash"
        ),
        hovertemplate=(
            "%{x}년<br>"
            "3차: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)


fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_9,
        mode="lines",
        name="9차 예측",
        line=dict(
            width=2,
            dash="dot"
        ),
        hovertemplate=(
            "%{x}년<br>"
            "9차: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)


fig_test.update_layout(
    title="2005년 이후 테스트 데이터",
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
# 2050년 예측 비교
# =========================================================
st.header("5. 2050년 예측 비교")

prediction_table = pd.DataFrame(
    {
        "모델": [
            "1차 (직선)",
            "3차 곡선",
            "9차 곡선"
        ],
        "2050년 예상 연평균기온 (°C)": [
            prediction_2050_1,
            prediction_2050_3,
            prediction_2050_9
        ]
    }
)


st.dataframe(
    prediction_table.style.format(
        {
            "2050년 예상 연평균기온 (°C)": "{:.2f}"
        }
    ),
    width="stretch",
    hide_index=True
)


# =========================================================
# 9차식 주의사항
# =========================================================
st.header("6. 왜 연도를 바꿔서 계산했을까?")

st.write(
    """
9차 다항회귀에서는 연도 자체를 그대로 사용하면
1900, 1950, 2000 같은 큰 숫자를 여러 번 거듭제곱하게 됩니다.

그래서 이 앱에서는 실제 연도를 그대로 회귀에 넣지 않고,

**x = (연도 - 2005) / 100**

으로 바꾸어 계산했습니다.

따라서 2005년은 x=0이 되고, 1955년은 x=-0.5,
2055년은 x=0.5가 됩니다.

그래프의 가로축은 이해하기 쉽도록 다시 실제 연도로 표시합니다.
"""
)


# =========================================================
# 데이터 기준
# =========================================================
with st.expander("📋 데이터 처리 기준"):
    st.write("• 2025년까지의 데이터만 사용")
    st.write("• 관측일수가 300일 미만인 연도는 제외")
    st.write("• 일별 평균기온을 평균하여 연평균기온 계산")
    st.write("• 2005년 이전: 훈련 데이터")
    st.write("• 2005년부터 2025년: 테스트 데이터")
    st.write("• 1차·3차·9차 모델 모두 훈련 데이터만 사용하여 학습")
    st.write("• 테스트 평균 오차 = 평균 절대 오차(MAE)")
    st.write("• 2050년 예측은 테스트 데이터와 무관하게 학습된 모델로 계산")
    st.write("• 다항회귀 계산 시 x = (연도 - 2005) / 100 사용")
