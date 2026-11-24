import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="تطبيق توقع المبيعات", layout="wide")


def generate_sample_data(days: int = 730, seed: int = 42) -> pd.DataFrame:
    """توليد بيانات مبيعات تجريبية لمدة سنتين"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    trend = np.linspace(500, 1200, days)
    weekly = np.where(dates.dayofweek >= 4, 200, -30)  # ارتفاع المبيعات في نهاية الأسبوع
    yearly = 150 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    noise = rng.normal(0, 60, size=days)

    sales = np.maximum(50, trend + weekly + yearly + noise).round(2)
    return pd.DataFrame({"date": dates, "sales": sales})


def preprocess_data(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """تنظيف البيانات وتجميعها يومياً"""
    clean = df[[date_col, target_col]].copy()
    clean[date_col] = pd.to_datetime(clean[date_col], errors="coerce")
    clean[target_col] = pd.to_numeric(clean[target_col], errors="coerce")
    clean = clean.dropna(subset=[date_col, target_col]).sort_values(date_col)

    clean["date"] = clean[date_col].dt.floor("D")
    daily = clean.groupby("date", as_index=False)[target_col].sum()
    daily = daily.rename(columns={target_col: "sales"})

    full_idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full_idx)
    daily["sales"] = daily["sales"].fillna(0.0)
    daily.index.name = "date"
    return daily.reset_index()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """استخراج خصائص التاريخ والزمن للنموذج"""
    out = df.copy()
    out["dayofweek"] = out["date"].dt.dayofweek
    out["dayofmonth"] = out["date"].dt.day
    out["dayofyear"] = out["date"].dt.dayofyear
    out["month"] = out["date"].dt.month
    out["is_weekend"] = (out["dayofweek"] >= 4).astype(int)
    out["time_idx"] = np.arange(len(out))
    return out


def forecast_sales(daily_df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """التنبؤ بالمبيعات المستقبلية باستخدام Random Forest"""
    history = daily_df.copy().reset_index(drop=True)
    for lag in [1, 7, 14, 28]:
        history[f"lag_{lag}"] = history["sales"].shift(lag)
    history["roll_mean_7"] = history["sales"].shift(1).rolling(7).mean()
    history["roll_mean_28"] = history["sales"].shift(1).rolling(28).mean()

    feat_df = add_features(history).dropna().reset_index(drop=True)
    feature_cols = [
        "dayofweek", "dayofmonth", "dayofyear", "month",
        "is_weekend", "time_idx", "lag_1", "lag_7", "lag_14",
        "lag_28", "roll_mean_7", "roll_mean_28"
    ]

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(feat_df[feature_cols], feat_df["sales"])

    train_preds = model.predict(feat_df[feature_cols])
    resid_std = float(np.std(feat_df["sales"] - train_preds))

    rows = []
    sim_series = list(daily_df["sales"].astype(float))
    last_date = daily_df["date"].max()
    base_idx = len(daily_df)

    for step in range(1, periods + 1):
        next_date = last_date + pd.Timedelta(days=step)
        idx = base_idx + step - 1

        def get_lag(k: int) -> float:
            pos = len(sim_series) - k
            return sim_series[pos] if pos >= 0 else sim_series[0]

        row_feat = pd.DataFrame([{
            "dayofweek": next_date.dayofweek,
            "dayofmonth": next_date.day,
            "dayofyear": next_date.dayofyear,
            "month": next_date.month,
            "is_weekend": int(next_date.dayofweek >= 4),
            "time_idx": idx,
            "lag_1": get_lag(1),
            "lag_7": get_lag(7),
            "lag_14": get_lag(14),
            "lag_28": get_lag(28),
            "roll_mean_7": float(np.mean(sim_series[-7:])),
            "roll_mean_28": float(np.mean(sim_series[-28:])),
        }])

        pred_val = max(0.0, float(model.predict(row_feat[feature_cols])[0]))
        sim_series.append(pred_val)
        band = 1.28 * resid_std * np.sqrt(1 + step / 30.0)
        rows.append({
            "date": next_date,
            "yhat": round(pred_val, 2),
            "yhat_lower": round(max(0.0, pred_val - band), 2),
            "yhat_upper": round(pred_val + band, 2),
        })

    return pd.DataFrame(rows)


def main():
    st.title("📊 تطبيق توقع المبيعات (Sales Forecasting)")
    st.write("ارفع ملف CSV يحتوي على التاريخ والمبيعات، أو جرب التطبيق مباشرة بالبيانات التجريبية.")

    # الشريط الجانبي للإعدادات
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        uploaded_file = st.file_uploader("ارفع ملف البيانات (CSV)", type=["csv"])
        horizon_days = st.slider("مدة التوقع (بالأيام)", min_value=7, max_value=180, value=30, step=7)

    # تحميل البيانات
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.subheader("تحديد الأعمدة")
        cols = list(raw_df.columns)
        col1, col2 = st.columns(2)
        date_col = col1.selectbox("عمود التاريخ (Date)", cols, index=0)
        default_num = next((i for i, c in enumerate(cols) if pd.api.types.is_numeric_dtype(raw_df[c])), min(1, len(cols) - 1))
        target_col = col2.selectbox("عمود المبيعات (Sales)", cols, index=default_num)
    else:
        raw_df = generate_sample_data()
        date_col, target_col = "date", "sales"
        st.info("ℹ️ يتم حالياً عرض بيانات تجريبية (730 يوم). يمكنك رفع ملفك الخاص من القائمة الجانبية.")

    daily_df = preprocess_data(raw_df, date_col, target_col)

    if len(daily_df) < 35:
        st.error("⚠️ يرجى توفير بيانات تاريخية لـ 35 يوماً على الأقل للحصول على توقعات دقيقة.")
        return

    # تشغيل التنبؤ
    future_df = forecast_sales(daily_df, horizon_days)

    # مؤشرات الأداء الرئيسية (KPIs)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("إجمالي المبيعات السابقة", f"{daily_df['sales'].sum():,.2f}")
    k2.metric("المتوسط اليومي السابق", f"{daily_df['sales'].mean():,.2f}")
    k3.metric(f"المبيعات المتوقعة ({horizon_days} يوم)", f"{future_df['yhat'].sum():,.2f}")
    k4.metric("المتوسط اليومي المتوقع", f"{future_df['yhat'].mean():,.2f}")

    # الرسم البياني التفاعلي
    fig = go.Figure()

    # مجال الثقة (الحد الأعلى والأدنى)
    fig.add_trace(go.Scatter(
        x=future_df["date"], y=future_df["yhat_upper"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=future_df["date"], y=future_df["yhat_lower"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(31, 119, 180, 0.2)", name="نطاق التوقع"
    ))

    # خط المبيعات الفعلية
    fig.add_trace(go.Scatter(
        x=daily_df["date"], y=daily_df["sales"],
        mode="lines", name="المبيعات الفعلية", line=dict(color="#2ca02c", width=2)
    ))

    # خط التوقعات
    fig.add_trace(go.Scatter(
        x=future_df["date"], y=future_df["yhat"],
        mode="lines", name="التوقعات المستقبلية", line=dict(color="#1f77b4", width=2.5, dash="dash")
    ))

    fig.update_layout(
        title="المبيعات الفعلية مقابل التوقعات المستقبلية",
        xaxis_title="التاريخ",
        yaxis_title="قيمة المبيعات",
        hovermode="x unified",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    # جدول النتائج وزر التحميل
    st.subheader("📋 جدول التوقعات التفصيلي")
    display_df = future_df.rename(columns={
        "date": "التاريخ",
        "yhat": "المبيعات المتوقعة",
        "yhat_lower": "الحد الأدنى",
        "yhat_upper": "الحد الأعلى"
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_buffer = io.StringIO()
    display_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 تحميل التوقعات بصيغة CSV",
        data=csv_buffer.getvalue(),
        file_name=f"sales_forecast_{horizon_days}_days.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
