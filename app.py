import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# CẤU HÌNH
# =========================
st.set_page_config(
    page_title="Dự báo giá cổ phiếu bằng AI",
    page_icon="📈",
    layout="wide"
)

# Giữ kết quả ổn định tương đối giống file gốc
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

st.title("☑ Ứng Dụng Dự Báo Giá Cổ Phiếu Bằng AI (LSTM)")
st.caption("Nhập mã cổ phiếu Việt Nam, ví dụ: VNM, FPT, HPG, MBB, VIC...")

# =========================
# NHẬP MÃ CỔ PHIẾU
# =========================
ticker_input = st.text_input(
    "Nhập mã cổ phiếu:",
    value="VNM",
    max_chars=10
).strip().upper()

# Tự động thêm .VN cho cổ phiếu Việt Nam
ticker = ticker_input if "." in ticker_input else ticker_input + ".VN"

forecast_days = st.slider(
    "Số phiên muốn dự báo:",
    min_value=5,
    max_value=15,
    value=10
)

run = st.button("🚀 Tải dữ liệu & Chạy dự báo", type="primary")


# =========================
# HÀM TẢI DỮ LIỆU
# =========================
@st.cache_data(ttl=3600)
def load_stock_data(ticker):
    df = yf.download(
        ticker,
        start="2018-01-01",
        end=pd.Timestamp.today().strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()

    # Xử lý trường hợp yfinance trả về MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            return pd.DataFrame()
    else:
        if "Close" not in df.columns:
            return pd.DataFrame()
        close = df["Close"]

    result = pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")})
    result = result.dropna()
    result.index = pd.to_datetime(result.index)

    return result


# =========================
# TẠO CHUỖI DỮ LIỆU
# =========================
def create_sequences(data, window_size=60):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


# =========================
# HUẤN LUYỆN + DỰ BÁO
# =========================
@st.cache_resource(show_spinner=False)
def train_lstm(ticker, forecast_days):
    df = load_stock_data(ticker)

    if len(df) < 150:
        raise ValueError(
            "Không đủ dữ liệu để huấn luyện. Hãy thử mã cổ phiếu có lịch sử giao dịch dài hơn."
        )

    data = df[["Close"]].values.astype(float)

    # 80% train - 20% test giống file gốc
    train_size = int(0.8 * len(data))

    train_data = data[:train_size]
    test_data = data[train_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    window_size = 60

    X_train, y_train = create_sequences(train_scaled, window_size)

    test_combined = np.vstack(
        (train_scaled[-window_size:], test_scaled)
    )
    X_test, y_test = create_sequences(test_combined, window_size)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Validation 10%
    val_size = max(1, int(0.1 * len(X_train)))

    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]

    # =========================
    # MÔ HÌNH LSTM
    # =========================
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_final,
        y_train_final,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )

    # =========================
    # DỰ BÁO TEST
    # =========================
    test_pred_scaled = model.predict(X_test, verbose=0)

    test_pred = scaler.inverse_transform(test_pred_scaled)
    y_test_real = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    )

    rmse = np.sqrt(
        mean_squared_error(y_test_real, test_pred)
    )
    mae = mean_absolute_error(
        y_test_real, test_pred
    )
    r2 = r2_score(
        y_test_real, test_pred
    )

    # =========================
    # DỰ BÁO TƯƠNG LAI
    # =========================
    last_60_days = np.vstack(
        (train_scaled, test_scaled)
    )[-window_size:]

    last_60_days = last_60_days.reshape(
        1, window_size, 1
    )

    future_predictions = []

    for _ in range(forecast_days):
        next_pred = model.predict(
            last_60_days,
            verbose=0
        )

        future_predictions.append(
            next_pred[0, 0]
        )

        last_60_days = np.append(
            last_60_days[:, 1:, :],
            next_pred.reshape(1, 1, 1),
            axis=1
        )

    future_predictions = np.array(
        future_predictions
    ).reshape(-1, 1)

    future_predictions = scaler.inverse_transform(
        future_predictions
    ).flatten()

    # Ngày giao dịch tương lai
    last_date = df.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days
    )

    return {
        "df": df,
        "model": model,
        "history": history,
        "test_pred": test_pred.flatten(),
        "y_test_real": y_test_real.flatten(),
        "future_predictions": future_predictions,
        "future_dates": future_dates,
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


# =========================
# GIAO DIỆN KẾT QUẢ
# =========================
if run:
    with st.spinner(
        f"Đang tải dữ liệu {ticker} và huấn luyện mô hình LSTM..."
    ):
        try:
            result = train_lstm(ticker, forecast_days)

            df = result["df"]
            future_predictions = result["future_predictions"]
            future_dates = result["future_dates"]

            current_price = float(df["Close"].iloc[-1])
            first_forecast = float(future_predictions[0])
            last_forecast = float(future_predictions[-1])

            change_pct = (
                (last_forecast - current_price)
                / current_price * 100
            )

            # =========================
            # THÔNG TIN TỔNG QUAN
            # =========================
            st.subheader(
                f"📊 Kết quả dự báo: {ticker_input}"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Giá đóng cửa gần nhất",
                    f"{current_price:,.0f} đ"
                )

            with col2:
                st.metric(
                    f"Dự báo phiên +1",
                    f"{first_forecast:,.0f} đ",
                    f"{(first_forecast-current_price)/current_price*100:+.2f}%"
                )

            with col3:
                st.metric(
                    f"Dự báo phiên +{forecast_days}",
                    f"{last_forecast:,.0f} đ",
                    f"{change_pct:+.2f}%"
                )

            with col4:
                if change_pct > 0:
                    st.success("📈 Xu hướng dự báo: TĂNG")
                elif change_pct < 0:
                    st.error("📉 Xu hướng dự báo: GIẢM")
                else:
                    st.info("➡️ Xu hướng dự báo: ĐI NGANG")

            # =========================
            # BIỂU ĐỒ CHÍNH
            # =========================
            st.subheader(
                "📈 Biểu đồ giá thật và dự báo AI"
            )

            fig, ax = plt.subplots(figsize=(14, 6))

            # Chỉ hiển thị phần lịch sử gần đây để biểu đồ dễ nhìn
            history_days = min(180, len(df))

            recent_df = df.iloc[-history_days:]

            ax.plot(
                recent_df.index,
                recent_df["Close"],
                label="Giá đóng cửa thực tế"
            )

            ax.plot(
                future_dates,
                future_predictions,
                marker="o",
                label="AI dự báo"
            )

            ax.axvline(
                df.index[-1],
                linestyle="--",
                alpha=0.7,
                label="Thời điểm hiện tại"
            )

            ax.set_title(
                f"Dự báo giá cổ phiếu {ticker_input} bằng LSTM"
            )
            ax.set_xlabel("Ngày")
            ax.set_ylabel("Giá (VNĐ)")
            ax.legend()
            ax.grid(alpha=0.3)

            st.pyplot(fig)

            # =========================
            # BẢNG DỰ BÁO
            # =========================
            st.subheader(
                f"🔮 Dự báo {forecast_days} phiên tiếp theo"
            )

            forecast_df = pd.DataFrame({
                "Ngày": future_dates.strftime("%d/%m/%Y"),
                "Giá dự báo (VNĐ)": [
                    round(x, 0)
                    for x in future_predictions
                ]
            })

            forecast_df["Thay đổi (%)"] = (
                (forecast_df["Giá dự báo (VNĐ)"] / current_price - 1)
                * 100
            ).round(2)

            st.dataframe(
                forecast_df,
                use_container_width=True,
                hide_index=True
            )

            # =========================
            # CHỈ SỐ MÔ HÌNH
            # =========================
            st.subheader("🧠 Đánh giá mô hình")

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "RMSE",
                    f"{result['rmse']:,.2f}"
                )

            with m2:
                st.metric(
                    "MAE",
                    f"{result['mae']:,.2f}"
                )

            with m3:
                st.metric(
                    "R²",
                    f"{result['r2']:.4f}"
                )

            # =========================
            # LOSS
            # =========================
            with st.expander(
                "Xem biểu đồ quá trình huấn luyện"
            ):
                fig2, ax2 = plt.subplots(figsize=(12, 5))

                ax2.plot(
                    result["history"].history["loss"],
                    label="Train Loss"
                )

                ax2.plot(
                    result["history"].history["val_loss"],
                    label="Validation Loss"
                )

                ax2.set_title(
                    "Loss Train vs Validation"
                )
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Loss")
                ax2.legend()
                ax2.grid(alpha=0.3)

                st.pyplot(fig2)

            st.warning(
                "⚠️ Đây là dự báo bằng mô hình LSTM dựa trên dữ liệu giá lịch sử, "
                "không phải tín hiệu mua/bán và không đảm bảo giá cổ phiếu sẽ tăng."
            )

        except Exception as e:
            st.error(
                f"Không thể chạy dự báo cho mã {ticker_input}: {e}"
            )
            st.info(
                "Hãy kiểm tra mã cổ phiếu. Với cổ phiếu Việt Nam, chỉ cần nhập "
                "mã như VNM, FPT, HPG, MBB, VIC; ứng dụng sẽ tự thêm .VN."
            )

else:
    st.info(
        "👆 Nhập mã cổ phiếu ở trên rồi bấm "
        "**Tải dữ liệu & Chạy dự báo**."
    )

    st.markdown("""
    ### Ví dụ
    - `VNM` → Vinamilk
    - `FPT` → FPT
    - `HPG` → Hòa Phát
    - `MBB` → MBBank
    - `VIC` → Vingroup

    Ứng dụng sẽ tự động tải dữ liệu từ Yahoo Finance, huấn luyện LSTM
    và hiển thị biểu đồ giá lịch sử + đường dự báo các phiên tiếp theo.
    """)
