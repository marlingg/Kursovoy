import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title='Predictive Maintenance Dashboard',
    page_icon='🔧',
    layout='wide'
)

# ─── Загрузка модели ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('model_artifacts.pkl', 'rb') as f:
        return pickle.load(f)

artifacts = load_artifacts()

model = artifacts['model']
scaler = artifacts['scaler']
le = artifacts['label_encoder']
FEATURES = artifacts['feature_cols']

# ─── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title('🔧 Параметри обладнання')

machine_id = st.sidebar.selectbox('Machine ID', list(range(1, 51)))
temperature = st.sidebar.slider('Температура (°C)', 40.0, 110.0, 75.0)
vibration = st.sidebar.slider('Вібрація (mm/s)', 0.0, 100.0, 30.0)
humidity = st.sidebar.slider('Вологість (%)', 10.0, 100.0, 55.0)
pressure = st.sidebar.slider('Тиск (бар)', 0.5, 10.0, 4.0)
energy = st.sidebar.slider('Споживання енергії (kW)', 0.0, 30.0, 10.0)
machine_status = st.sidebar.selectbox('Статус машини', [0, 1, 2])
anomaly_flag = st.sidebar.checkbox('Прапорець аномалії', value=False)
pred_life = st.sidebar.number_input('Залишковий ресурс (год)', 0, 1000, 200)
downtime_risk = st.sidebar.slider('Ризик простою', 0.0, 1.0, 0.2)
failure_type = st.sidebar.selectbox('Тип відмови', artifacts['failure_types'])

predict_btn = st.sidebar.button('🚀 Прогнозувати', use_container_width=True)

# ─── Main ───────────────────────────────────────────────────────────────────
st.title('🏭 Predictive Maintenance — IoT Dashboard')
st.markdown('Прогнозування потреби в технічному обслуговуванні виробничого обладнання')

if predict_btn:

    failure_enc = le.transform([failure_type])[0]

    row = {
        'machine_id': machine_id,
        'temperature': temperature,
        'vibration': vibration,
        'humidity': humidity,
        'pressure': pressure,
        'energy_consumption': energy,
        'machine_status': machine_status,
        'anomaly_flag': int(anomaly_flag),
        'predicted_remaining_life': pred_life,
        'downtime_risk': downtime_risk,
        'failure_type_enc': failure_enc,
        'hour': 12,
        'dayofweek': 1,
        'month': 5,
        'temperature_roll10': temperature,
        'vibration_roll10': vibration,
        'energy_consumption_roll10': energy,
        'temperature_lag1': temperature,
        'vibration_lag1': vibration,
        'temp_x_vib': temperature * vibration,
        'energy_per_pressure': energy / (pressure + 1e-6),
    }

    X_input = pd.DataFrame([row])[FEATURES]

    prob = model.predict_proba(X_input)[0, 1]
    pred = int(prob >= 0.5)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            '🔮 Прогноз',
            'ПОТРІБНЕ' if pred else 'НЕ ПОТРІБНЕ',
            delta='Обслуговування' if pred else 'Продовжити роботу'
        )

    with col2:
        st.metric('📊 Ймовірність', f'{prob:.1%}')

    with col3:
        risk_label = 'Критичний' if prob > 0.7 else ('Помірний' if prob > 0.4 else 'Низький')
        st.metric('⚠️ Рівень ризику', risk_label)

    # ─── Gauge chart ─────────────────────────────────────────────────────────
    fig_gauge = go.Figure(go.Indicator(
        mode='gauge+number',
        value=prob * 100,
        title={'text': 'Ймовірність обслуговування (%)'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#e74c3c' if prob > 0.5 else '#2ecc71'},
            'steps': [
                {'range': [0, 40], 'color': '#d5f5e3'},
                {'range': [40, 70], 'color': '#fef9e7'},
                {'range': [70, 100], 'color': '#fadbd8'}
            ],
            'threshold': {'line': {'color': 'red', 'width': 4}, 'value': 50}
        }
    ))

    st.plotly_chart(fig_gauge, use_container_width=True)

    # ─── Таблица сигналов ────────────────────────────────────────────────────
    signals = pd.DataFrame([
        {'Параметр': 'Температура', 'Значення': f'{temperature:.1f} °C', 'Статус': '🔴' if temperature > 90 else '🟢'},
        {'Параметр': 'Вібрація', 'Значення': f'{vibration:.1f} mm/s', 'Статус': '🔴' if vibration > 70 else '🟢'},
        {'Параметр': 'Тиск', 'Значення': f'{pressure:.2f} бар', 'Статус': '🟡' if pressure < 1.5 else '🟢'},
        {'Параметр': 'Аномалія', 'Значення': str(bool(anomaly_flag)), 'Статус': '🔴' if anomaly_flag else '🟢'},
        {'Параметр': 'Ризик простою', 'Значення': f'{downtime_risk:.2f}', 'Статус': '🔴' if downtime_risk > 0.7 else '🟢'},
    ])

    st.markdown('### 📋 Таблиця сигналів')
    st.dataframe(signals, use_container_width=True)

else:
    st.info('👈 Введіть параметри та натисніть «Прогнозувати»')
