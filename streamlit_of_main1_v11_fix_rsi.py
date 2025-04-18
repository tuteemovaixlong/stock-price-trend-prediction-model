import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import streamlit as st

# Function to calculate RSI
def calculate_rsi(data, period=14): #period là chu kỳ để tính
    delta = data['Close'].diff()  # Tính sự thay đổi giá Close
    gain = delta.where(delta > 0, 0).fillna(0)  # Lấy phần dương của delta (gain)
    # Tính toán giá trị tăng (gain) bằng cách lấy các giá trị dương trong delta và thay
    # thế các giá trị âm bằng 0. Các giá trị NaN (nếu có) được điền bằng 0.
    loss = -delta.where(delta < 0, 0).fillna(0)  # Lấy phần âm của delta và chuyển thành dương (loss)
    # Tính toán giá trị giảm (loss) bằng cách lấy các giá trị âm trong delta,
    # đổi dấu thành dương và thay thế các giá trị dương bằng 0. Các giá trị NaN (nếu có) được điền bằng 0.
    avg_gain = gain.rolling(window=period, min_periods=period).mean()  # Trung bình gain
    avg_loss = loss.rolling(window=period, min_periods=period).mean()  # Trung bình loss

    # Tránh chia cho 0 bằng cách thay thế avg_loss = 0 bằng NaN
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Nếu avg_loss = 0, RS = NaN
    # Tính tỷ lệ RS (Relative Strength) bằng cách chia avg_gain cho avg_loss.
    # Nếu avg_loss bằng 0, thay thế bằng NaN để tránh chia cho 0.

    rsi = 100 - (100 / (1 + rs))  # Công thức RSI
    return rsi.fillna(0)  # Điền NaN bằng 0 để tránh lỗi


def build_model(input_shape):
    model = Sequential()
# Tạo một mô hình tuần tự (sequential model) sử dụng lớp Sequential của Keras.
# Mô hình tuần tự là một cách đơn giản để xây dựng các mạng neural, trong đó các lớp được xếp chồng lên nhau theo thứ tự.
    model.add(Dense(units=512, activation='tanh', input_shape=(input_shape,), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(units=256, activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(units=128, activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    )
    return model


def train_model(x_train_arr, y_train_arr, num_iterations):
    model = build_model(x_train_arr.shape[1])
    history = model.fit(x_train_arr, y_train_arr, epochs=num_iterations, batch_size=32, validation_split=0.2)
    return model, history

# Function to evaluate the model


def evaluate(model, x_set, y_set, mode):
    y_predict = model.predict(x_set)
    count_correct = 0
    count_time = 0
    for i in range(y_set.shape[0]):
        trueTrendRate = y_set[i]
        predictTrendRate = y_predict[i]
        count_time += 1
        if ((trueTrendRate > 0.1) and (predictTrendRate > 0.1)) or (
                (trueTrendRate <= 0.1) and (predictTrendRate <= 0.1)):
            count_correct += 1
    correct_rate = count_correct / count_time
    st.write(f"{mode} correct rate: {correct_rate * 100:.2f}%")

# Streamlit UI
st.title("Mô hình dự đoán xu hướng giá cổ phiếu bằng mạng neural nhân tạo ANN")

# Sử dụng session_state để lưu dữ liệu, mô hình đã huấn luyện, trạng thái

if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'history' not in st.session_state:
    st.session_state['history'] = None
if 'input_open' not in st.session_state:
    st.session_state['input_open'] = 0.0
if 'input_close' not in st.session_state:
    st.session_state['input_close'] = 0.0
if 'input_low' not in st.session_state:
    st.session_state['input_low'] = 0.0
if 'input_high' not in st.session_state:
    st.session_state['input_high'] = 0.0

uploaded_file = st.file_uploader("Tải lên file CSV dữ liệu", type="csv")

# Nếu có file tải lên, lưu trữ vào session_state
if uploaded_file is not None:
    if st.session_state['data'] is None:
        st.session_state['data'] = pd.read_csv(uploaded_file)
        st.write("File đã được tải lên và dữ liệu đã được lưu.")

# Kiểm tra nếu dữ liệu đã có sẵn trong session_state
if st.session_state['data'] is not None:
    data = st.session_state['data']

    # Khởi tạo button
    part1_button = st.button("Thực hiện xử lý dữ liệu lần 1 và tính toán biểu đồ")
    part2_button = st.button("Thực hiện huấn luyện mô hình và dự đoán")

    if part1_button:
        columns_to_clean = ['Open', 'Close', 'Low', 'High']
        for column in columns_to_clean:
            if data[column].dtype == 'object':
                data[column] = data[column].str.replace('$', '').astype(float)
            else:
                data[column] = data[column].astype(float)

        data['RSI'] = calculate_rsi(data)

        st.write("Chỉ số RSI:")
        fig_rsi, ax_rsi = plt.subplots()
        ax_rsi.plot(data.index, data['RSI'], label='RSI', color='blue')
        ax_rsi.axhline(30, linestyle='--', color='r', label="Oversold (30)")
        ax_rsi.axhline(70, linestyle='--', color='g', label="Overbought (70)")
        ax_rsi.set_xlabel('Thời gian')
        ax_rsi.set_ylabel('RSI')
        ax_rsi.legend()
        st.pyplot(fig_rsi)

        st.write("Biểu đồ giá đóng cửa:")
        fig_price, ax_price = plt.subplots()
        ax_price.plot(data.index, data['Close'], label='Close Price', color='green')
        ax_price.set_xlabel('Thời gian')
        ax_price.set_ylabel('Giá đóng cửa')
        ax_price.legend()
        st.pyplot(fig_price)

    scaler_open = MinMaxScaler()
    scaler_close = MinMaxScaler()
    scaler_low = MinMaxScaler()
    scaler_high = MinMaxScaler()
    scaler_rsi = MinMaxScaler()
    if part2_button:
        # Phần 2: Huấn luyện mô hình nếu chưa huấn luyện
        if st.session_state['model'] is None:
            columns_to_clean = ['Open', 'Close', 'Low', 'High']

            # Chuyển đổi dữ liệu thành số và xử lý giá trị thiếu
            for column in columns_to_clean:
                data[column] = pd.to_numeric(data[column], errors='coerce')
                data[column] = data[column].fillna(0)  # Điền NaN bằng 0

            # Tính toán RSI
            data['RSI'] = calculate_rsi(data)



            data[['Open']] = scaler_open.fit_transform(data[['Open']])
            data[['Close']] = scaler_close.fit_transform(data[['Close']])
            data[['Low']] = scaler_low.fit_transform(data[['Low']])
            data[['High']] = scaler_high.fit_transform(data[['High']])
            data[['RSI']] = scaler_rsi.fit_transform(data[['RSI']])

            arr = data.to_numpy()

            # Parameters
            trainSize = 2200
            testSize = 315
            historyDays = 50
            num_iterations = 157

            x_train, y_train, x_test, y_test = [], [], [], []
            for i, rec in enumerate(arr):
                if i < historyDays:
                    continue
                x_i = []
                for j in reversed(range(historyDays)):
                    x_i.extend(arr[i - j - 1][2:7])
                y_i = [arr[i][3] / arr[i - 1][3] if arr[i - 1][3] != 0 else 0]
                if i <= trainSize + historyDays - 1:
                    x_train.append(x_i)
                    y_train.append(y_i)
                elif i <= trainSize + testSize + historyDays - 1:
                    x_test.append(x_i)
                    y_test.append(y_i)

            x_train_arr = np.array(x_train)
            y_train_arr = np.array(y_train)
            x_test_arr = np.array(x_test)
            y_test_arr = np.array(y_test)

            # Huấn luyện mô hình
            st.session_state['model'], st.session_state['history'] = train_model(x_train_arr, y_train_arr, num_iterations)

            # Đánh giá trên dữ liệu huấn luyện và kiểm thử
            st.write("Kết quả huấn luyện:")
            evaluate(st.session_state['model'], x_train_arr, y_train_arr, "Train")
            st.write("Kết quả kiểm thử:")
            evaluate(st.session_state['model'], x_test_arr, y_test_arr, "Test")

            # Vẽ biểu đồ hàm lỗi
            st.write("Quá trình suy giảm của hàm lỗi:")
            fig, ax = plt.subplots()
            ax.plot(st.session_state['history'].history['loss'], label='Training Loss')
            ax.plot(st.session_state['history'].history['val_loss'], label='Validation Loss')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()
            st.pyplot(fig)

        else:
            st.write("Mô hình đã được huấn luyện trước đó.")

    # Dự đoán xu hướng (sau khi mô hình đã huấn luyện)
    st.subheader("Mời bạn nhập dữ liệu để thực hiện dự đoán xu hướng giá của cổ phiếu")

    # Sử dụng session_state để giữ lại giá trị của các input
    st.session_state['input_open'] = st.number_input("Nhập giá Open", min_value=0.0, value=st.session_state['input_open'], format="%.2f")
    st.session_state['input_close'] = st.number_input("Nhập giá Close", min_value=0.0, value=st.session_state['input_close'], format="%.2f")
    st.session_state['input_low'] = st.number_input("Nhập giá Low", min_value=0.0, value=st.session_state['input_low'], format="%.2f")
    st.session_state['input_high'] = st.number_input("Nhập giá High", min_value=0.0, value=st.session_state['input_high'], format="%.2f")

    # Button to calculate RSI and make prediction

    if st.button("Tính RSI và dự đoán"):
        if st.session_state['model'] is None:
            st.write("Vui lòng huấn luyện mô hình trước khi dự đoán.")
        else:
            input_open = st.session_state['input_open']
            input_close = st.session_state['input_close']
            input_low = st.session_state['input_low']
            input_high = st.session_state['input_high']

            input_delta = input_close - input_open
            input_gain = max(input_delta, 0)
            input_loss = abs(min(input_delta, 0))

            avg_gain = input_gain
            avg_loss = input_loss
            if avg_loss == 0:
                input_rsi = 100
            else:
                rs = avg_gain / avg_loss
                input_rsi = 100 - (100 / (1 + rs))

            historyDays = 50
            if len(data) >= historyDays:
                recent_data = data[-historyDays:][['Open', 'Close', 'Low', 'High', 'RSI']].values
                input_data = np.array([[input_open, input_close, input_low, input_high, input_rsi]])

                if len(data) >= historyDays:

                    recent_data = data[-historyDays:][['Open', 'Close', 'Low', 'High', 'RSI']].values

                    scaler_open.fit(recent_data[:, 0].reshape(-1, 1))
                    scaler_close.fit(recent_data[:, 1].reshape(-1, 1))
                    scaler_low.fit(recent_data[:, 2].reshape(-1, 1))
                    scaler_high.fit(recent_data[:, 3].reshape(-1, 1))
                    scaler_rsi.fit(recent_data[:, 4].reshape(-1, 1))

                    input_data = recent_data.flatten().reshape(1, -1)

                    prediction = st.session_state['model'].predict(input_data)
                    if prediction[0] > 0.5:
                        st.write("Dự đoán: Giá sẽ tăng")
                    else:
                        st.write("Dự đoán: Giá sẽ giảm")
                else:
                    st.write("Không đủ dữ liệu lịch sử để thực hiện dự đoán.")

else:
    st.write("Vui lòng tải lên file CSV dữ liệu.")
