# app.py — Predictive Maintenance Web Service
# Запуск: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─── Конфігурація сторінки ───────────────────────────────────────────────────
st.set_page_config(
    page_title='Predictive Maintenance System',
        layout='wide',
    initial_sidebar_state='expanded'
)

# ─── CSS-стилі ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header{
    font-size:36px;
    font-weight:700;
    color:#111827;
    padding-bottom:10px;
}
.section-title{
    font-size:22px;
    font-weight:600;
    color:#1f2937;
}
div[data-testid="stMetric"]{
    border:1px solid #d1d5db;
    border-radius:8px;
    padding:10px;
    background:white;
}
.stButton > button{
    width:100%;
    background:#1f2937;
    color:white;
    border:none;
    border-radius:6px;
    font-weight:600;
    height:45px;
}
.stButton > button:hover{
    background:#111827;
}
[data-testid="stSidebar"]{
    border-right:1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)


# ─── Завантаження моделі ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Завантаження моделі...')
def load_artifacts(path='model_artifacts.pkl'):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


artifacts = load_artifacts()

if artifacts is None:
    st.error(
        '❌ Файл `model_artifacts.pkl` не знайдено. '
        'Переконайтесь, що він знаходиться в одній папці з `app.py`.'
    )
    st.stop()

model          = artifacts['model']
scaler         = artifacts['scaler']
le             = artifacts['label_encoder']
FEATURES       = artifacts['feature_cols']
FAILURE_TYPES  = artifacts['failure_types']


# ─── Допоміжні функції ───────────────────────────────────────────────────────
def build_input_row(params: dict) -> pd.DataFrame:
    """Формує DataFrame з одного запису для передачі в модель."""
    now = datetime.now()
    row = {
        'machine_id':                  params['machine_id'],
        'temperature':                 params['temperature'],
        'vibration':                   params['vibration'],
        'humidity':                    params['humidity'],
        'pressure':                    params['pressure'],
        'energy_consumption':          params['energy'],
        'machine_status':              params['machine_status'],
        'anomaly_flag':                int(params['anomaly_flag']),
        'predicted_remaining_life':    params['pred_life'],
        'downtime_risk':               params['downtime_risk'],
        'failure_type_enc':            le.transform([params['failure_type']])[0],
        'hour':                        now.hour,
        'dayofweek':                   now.weekday(),
        'month':                       now.month,
        # rolling / lag — при одиничному вводі = поточне значення
        'temperature_roll10':          params['temperature'],
        'vibration_roll10':            params['vibration'],
        'energy_consumption_roll10':   params['energy'],
        'temperature_lag1':            params['temperature'],
        'vibration_lag1':              params['vibration'],
        # похідні ознаки
        'temp_x_vib':                  params['temperature'] * params['vibration'],
        'energy_per_pressure':         params['energy'] / (params['pressure'] + 1e-6),
    }
    return pd.DataFrame([row])[FEATURES]


def risk_level(prob: float) -> tuple[str, str]:
    """Повертає (мітка, css-клас) для рівня ризику."""
    if prob >= 0.70:
        return 'Високий', 'alert-critical'
    if prob >= 0.40:
        return 'Середній', 'alert-warning'
    return 'Низький', 'alert-ok'


def signal_status(value, high=None, low=None) -> str:
    if high is not None and value > high:
        return 'Перевищення'
    if low is not None and value < low:
        return 'Нижче допустимого рівня'
    return 'Норма'


# ─── Бічна панель (введення параметрів) ─────────────────────────────────────
st.sidebar.markdown('## Параметри обладнання')
st.sidebar.markdown('---')

machine_id     = st.sidebar.selectbox('Обладнання', list(range(1, 51)), index=0)
st.sidebar.markdown('**Сенсорні показники**')
temperature    = st.sidebar.slider('Температура (°C)',    40.0, 115.0, 75.0, 0.5)
vibration      = st.sidebar.slider('Вібрація (mm/s)',      0.0, 100.0, 30.0, 0.5)
humidity       = st.sidebar.slider('Вологість (%)',        10.0, 100.0, 55.0, 1.0)
pressure       = st.sidebar.slider('Тиск (бар)',           0.5,  10.0,  4.0, 0.1)
energy         = st.sidebar.slider('Спожита енергія (kW)',  0.0,  30.0, 10.0, 0.5)

st.sidebar.markdown('**Стан обладнання**')
machine_status = st.sidebar.selectbox('Статус машини', [0, 1, 2],
                                       format_func=lambda x: {0:'Зупинена', 1:'Активна', 2:'Режим тех. обсл.'}[x])
anomaly_flag   = st.sidebar.checkbox('Прапорець аномалії', value=False)
pred_life      = st.sidebar.number_input('Залишковий ресурс (год)', 0, 1000, 200, step=10)
downtime_risk  = st.sidebar.slider('Ризик простою (0–1)', 0.0, 1.0, 0.2, 0.01)
failure_type   = st.sidebar.selectbox('Тип відмови', FAILURE_TYPES)

st.sidebar.markdown('---')
predict_btn = st.sidebar.button('Виконати прогноз')

# ─── Головна панель ──────────────────────────────────────────────────────────
st.markdown('<p class="main-header">Predictive Maintenance System</p>',
            unsafe_allow_html=True)
st.markdown('Прогнозування потреби в технічному обслуговуванні виробничого обладнання на основі IoT-сенсорів')
st.markdown('---')

# ─── Вкладки інтерфейсу ──────────────────────────────────────────────────────
tab_pred, tab_analysis, tab_info = st.tabs(
    ['Прогнозування', 'Аналітика', 'Документація']
)

# ════════════════════════════════════════════════════════
# Вкладка 1 — Прогноз
# ════════════════════════════════════════════════════════
with tab_pred:
    if not predict_btn:
        st.info('👈 Введіть параметри обладнання в бічній панелі та натисніть **«Виконати прогноз»**')
        col_a, col_b, col_c = st.columns(3)
        col_a.markdown('**Сервіс виконує:**')
        col_a.markdown('- 🎯 Прогноз ймовірності обслуговування\n- Gauge-індикатор ризику\n- Таблиця сигналів')
        col_b.markdown('**Вхідні дані:**')
        col_b.markdown('- Температура, вібрація, вологість\n- Тиск, споживання енергії\n- Статус, аномалії, тип відмови')
        col_c.markdown('**Модель:**')
        col_c.markdown('- RandomForestClassifier\n- 21 ознака (з rolling & lag)\n- Accuracy ≈ 1.0 / ROC-AUC ≈ 1.0')

    else:
        params = dict(
            machine_id=machine_id, temperature=temperature, vibration=vibration,
            humidity=humidity, pressure=pressure, energy=energy,
            machine_status=machine_status, anomaly_flag=anomaly_flag,
            pred_life=pred_life, downtime_risk=downtime_risk, failure_type=failure_type
        )

        X_input = build_input_row(params)
        prob    = float(model.predict_proba(X_input)[0, 1])
        pred    = int(prob >= 0.5)
        risk_lbl, risk_css = risk_level(prob)

        # ── Метрики ────────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                'Статус обслуговування',
                'ПОТРІБНЕ' if pred else 'НЕ ПОТРІБНЕ',
                delta='Терміново!' if pred else 'Продовжити роботу',
                delta_color='inverse'
            )
        with col2:
            st.metric('Ймовірність відмови', f'{prob:.1%}')
        with col3:
            st.metric('Категорія ризику', risk_lbl.split()[-1])
        with col4:
            st.metric('Обладнання', f'Машина №{machine_id}')

        st.markdown('---')

        # ── Gauge chart ────────────────────────────────────────────────────
        col_g, col_t = st.columns([1, 1])
        with col_g:
            bar_color = '#e53935' if prob > 0.5 else ('#fb8c00' if prob > 0.35 else '#43a047')
            fig_gauge = go.Figure(go.Indicator(
                mode='gauge+number+delta',
                value=round(prob * 100, 1),
                delta={'reference': 50, 'increasing': {'color': '#e53935'},
                       'decreasing': {'color': '#43a047'}},
                number={'suffix': '%', 'font': {'size': 36}},
                title={'text': 'Ймовірність обслуговування', 'font': {'size': 15}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': bar_color, 'thickness': 0.25},
                    'bgcolor': 'white',
                    'borderwidth': 1,
                    'steps': [
                        {'range': [0,  40], 'color': '#e8f5e9'},
                        {'range': [40, 70], 'color': '#fff3e0'},
                        {'range': [70, 100],'color': '#ffebee'},
                    ],
                    'threshold': {
                        'line': {'color': '#1a73e8', 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=40, b=20, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Таблиця сигналів ───────────────────────────────────────────────
        with col_t:
            st.markdown('#### Таблиця сигналів')
            signals = pd.DataFrame([
                {'Параметр': 'Температура',
                 'Значення': f'{temperature:.1f} °C',
                 'Норма': '40–90 °C',
                 'Статус': signal_status(temperature, high=90)},
                {'Параметр': 'Вібрація',
                 'Значення': f'{vibration:.1f} mm/s',
                 'Норма': '0–70 mm/s',
                 'Статус': signal_status(vibration, high=70)},
                {'Параметр': 'Тиск',
                 'Значення': f'{pressure:.2f} бар',
                 'Норма': '1.5–8 бар',
                 'Статус': signal_status(pressure, low=1.5)},
                {'Параметр': 'Енергія',
                 'Значення': f'{energy:.1f} kW',
                 'Норма': '0–25 kW',
                 'Статус': signal_status(energy, high=25)},
                {'Параметр': 'Аномалія',
                 'Значення': 'ТАК' if anomaly_flag else 'НІ',
                 'Норма': 'Немає',
                 'Статус': '🔴 Виявлено' if anomaly_flag else '🟢 Норма'},
                {'Параметр': 'Ризик простою',
                 'Значення': f'{downtime_risk:.2f}',
                 'Норма': '0–0.7',
                 'Статус': signal_status(downtime_risk, high=0.7)},
                {'Параметр': 'Залишковий ресурс',
                 'Значення': f'{pred_life} год',
                 'Норма': '>50 год',
                 'Статус': signal_status(pred_life, low=50)},
            ])
            st.dataframe(signals, use_container_width=True, hide_index=True)

        # ── Рекомендація ───────────────────────────────────────────────────
        st.markdown('---')
        st.markdown('#### Рекомендації щодо технічного обслуговування')
        if prob >= 0.70:
            st.error(
                f'**Негайне обслуговування!** Ймовірність відмови: **{prob:.1%}**. '
                f'Машина №{machine_id} потребує термінового втручання. '
                f'Залишковий ресурс: {pred_life} год. Тип відмови: {failure_type}.'
            )
        elif prob >= 0.40:
            st.warning(
                f'**Заплануйте обслуговування.** Ймовірність: **{prob:.1%}**. '
                f'Рекомендується провести діагностику машини №{machine_id} '
                f'протягом найближчих {max(1, pred_life // 24)} діб.'
            )
        else:
            st.success(
                f'**Обслуговування не потрібне.** Ймовірність: **{prob:.1%}**. '
                f'Машина №{machine_id} працює в штатному режимі. '
                f'Наступна перевірка рекомендована через {pred_life} год.'
            )

# ════════════════════════════════════════════════════════
# Вкладка 2 — Аналіз параметрів
# ════════════════════════════════════════════════════════
with tab_analysis:
    st.markdown('### Візуалізація поточних параметрів відносно норм')

    params_analysis = {
        'Температура (°C)':   (temperature,   40,  90,  115),
        'Вібрація (mm/s)':    (vibration,      0,  70,  100),
        'Вологість (%%)':     (humidity,       10,  90,  100),
        'Тиск (бар)':         (pressure,       1.5, 8,   10),
        'Енергія (kW)':       (energy,         0,  25,   30),
        'Ризик простою':      (downtime_risk,  0,  0.7,   1),
    }

    col1, col2 = st.columns(2)
    cols = [col1, col2]
    for i, (label, (val, vmin, vmax, vabs)) in enumerate(params_analysis.items()):
        pct = (val - vmin) / (vabs - vmin) * 100 if vabs != vmin else 0
        norm_pct = (vmax - vmin) / (vabs - vmin) * 100
        color = '#e53935' if val > vmax else ('#fb8c00' if val > vmax * 0.85 else '#43a047')

        fig = go.Figure(go.Bar(
            x=[pct], y=[label], orientation='h',
            marker_color=color, width=0.5
        ))
        fig.add_shape(type='line', x0=norm_pct, x1=norm_pct, y0=-0.4, y1=0.4,
                      line=dict(color='#1a73e8', width=2, dash='dash'))
        fig.update_layout(
            height=100, margin=dict(l=0, r=10, t=25, b=0),
            xaxis=dict(range=[0, 100], title='% від максимуму', showgrid=False),
            yaxis=dict(showticklabels=True),
            title=dict(text=f'{label}: <b>{val}</b>', font=dict(size=13)),
            showlegend=False
        )
        cols[i % 2].plotly_chart(fig, use_container_width=True)

    # Spider / Radar chart
    st.markdown('---')
    st.markdown('### Радарна карта стану обладнання')
    categories = ['Температура', 'Вібрація', 'Вологість', 'Тиск', 'Енергія', 'Ризик простою']
    norm_vals = [
        temperature / 115 * 100,
        vibration   / 100 * 100,
        humidity    / 100 * 100,
        pressure    / 10  * 100,
        energy      / 30  * 100,
        downtime_risk     * 100,
    ]
    thresholds = [90/115*100, 70/100*100, 90, 80, 25/30*100, 70]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=norm_vals + [norm_vals[0]],
        theta=categories + [categories[0]],
        fill='toself', name='Поточні значення',
        line=dict(color='#1a73e8'), fillcolor='rgba(26,115,232,0.2)'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=thresholds + [thresholds[0]],
        theta=categories + [categories[0]],
        fill='toself', name='Порогові значення',
        line=dict(color='#e53935', dash='dash'), fillcolor='rgba(229,57,53,0.1)'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=420, showlegend=True,
        title='Порівняння поточних значень із пороговими нормами'
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ════════════════════════════════════════════════════════
# Вкладка 3 — Про модель
# ════════════════════════════════════════════════════════
with tab_info:
    st.markdown('### Інформація про модель та набори даних')

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""
**Набори даних:**
- Smart Manufacturing IoT-Cloud Monitoring Dataset (Kaggle, 2025-03-03)
- Real-Time IoT-Driven Production System Dataset (Kaggle, 2025-03-10)
- Загальний розмір після об'єднання: **102 460 записів**

**Модель:**
- Алгоритм: **RandomForestClassifier**
- Кількість ознак: **21** (вихідні + rolling + lag + похідні)
- Цільова змінна: `maintenance_required` (0/1)

**Метрики на тесті:**
- Accuracy ≈ **1.000**
- ROC-AUC ≈ **1.000**
- RMSE ≈ **0.000**
""")
    with col_i2:
        st.markdown("""
**Ключові ознаки (ТОП-5):**
1. `machine_status` — статус машини (~35% важливості)
2. `anomaly_flag` — прапорець аномалії (~16%)
3. `predicted_remaining_life` — залишковий ресурс (~16%)
4. `failure_type_enc` — тип відмови (~10%)
5. `downtime_risk` — ризик простою (~10%)

**Технічний стек:**
- Python, Pandas, NumPy, Scikit-learn
- XGBoost, LightGBM, Plotly, Streamlit

**Структура проєкту:**
```
app.py                  ← вебсервіс (цей файл)
model_artifacts.pkl     ← модель + артефакти
main_dataset.csv        ← об'єднаний датасет
predictive_maintenance.ipynb  ← notebook
requirements.txt
```
""")

    st.markdown('---')
    st.markdown('**Схема попереднього оброблення:**')
    proc_steps = pd.DataFrame([
        {'Крок': '1. Нормалізація ID',    'Деталі': 'M001→1, M002→2 … числовий int64'},
        {'Крок': '2. Імпутація пропусків','Деталі': 'Медіана для числових; «Unknown» для failure_type'},
        {'Крок': '3. Кодування',          'Деталі': 'LabelEncoder для failure_type'},
        {'Крок': '4. Часові ознаки',      'Деталі': 'hour, dayofweek, month з timestamp'},
        {'Крок': '5. Rolling-ознаки',     'Деталі': 'rolling(10).mean() для temp, vib, energy по machine_id'},
        {'Крок': '6. Лагові ознаки',      'Деталі': 'lag(1) для temperature, vibration'},
        {'Крок': '7. Похідні ознаки',     'Деталі': 'temp×vib, energy/pressure'},
        {'Крок': '8. Масштабування',      'Деталі': 'StandardScaler (для лін. моделей)'},
    ])
    st.dataframe(proc_steps, use_container_width=True, hide_index=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown('---')
st.markdown(
    '<small>Predictive Maintenance IoT Dashboard | '
    'RandomForestClassifier | Streamlit</small>',
    unsafe_allow_html=True
)
