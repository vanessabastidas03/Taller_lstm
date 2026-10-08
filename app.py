from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
RANGOS = {
    'demanda_mw': (0, 1500),
    'temperatura_c': (-10, 50),
    'humedad_pct': (0, 100),
    'viento_kmh': (0, 100),
    'radiacion_wm2': (0, 1500),
    'precipitacion_mm': (0, 100),
    'precio_kwh': (0, 100),
    'hora': (0, 23),
    'dia_semana': (0, 6),
    'fin_semana': (0, 1),
    'festivo': (0, 1),
    'mes': (1, 12),
}

st.set_page_config(page_title='Predicción de demanda energética', layout='wide')

@st.cache_resource
def cargar():
    cfg = joblib.load(str(BASE_DIR / 'modelos' / 'config.joblib'))
    ruta_modelo = Path(cfg['ruta_modelo'])
    if not ruta_modelo.is_absolute():
        ruta_modelo = BASE_DIR / ruta_modelo
    modelo = load_model(str(ruta_modelo))
    sx = joblib.load(str(BASE_DIR / 'modelos' / 'scaler_x.joblib'))
    sy = joblib.load(str(BASE_DIR / 'modelos' / 'scaler_y.joblib'))
    return cfg, modelo, sx, sy

# Validación rápida antes de predecir para asegurar entradas reales y coherentes.
def validar(d):
    problemas = []
    for col, (min_v, max_v) in RANGOS.items():
        if col not in d.columns:
            continue

        nulos = d[col].isna()
        if nulos.any():
            filas = sorted(nulos[nulos].index.tolist())
            problemas.append(f"La columna '{col}' tiene celdas vacías en filas: {filas}.")

        mask = d[col].notna()
        fuera = d.loc[mask, col][(d.loc[mask, col] < min_v) | (d.loc[mask, col] > max_v)]
        if not fuera.empty:
            filas = sorted(fuera.index.tolist())
            problemas.append(
                f"La columna '{col}' está fuera del rango válido [{min_v}, {max_v}] en filas: {filas}."
            )
    return problemas


def agregar_ciclicas(d):
    d = d.copy()
    d['hora_sin'] = np.sin(2*np.pi*d['hora']/24)
    d['hora_cos'] = np.cos(2*np.pi*d['hora']/24)
    d['dia_sin']  = np.sin(2*np.pi*d['dia_semana']/7)
    d['dia_cos']  = np.cos(2*np.pi*d['dia_semana']/7)
    d['mes_sin']  = np.sin(2*np.pi*d['mes']/12)
    d['mes_cos']  = np.cos(2*np.pi*d['mes']/12)
    return d

cfg, modelo, sx, sy = cargar()
n, features = cfg['n'], cfg['features']

st.title('Predicción de la demanda eléctrica de la próxima hora')
st.write(f'El modelo usa las últimas {n} horas (la última fila es la hora más reciente). '
         'Edite los valores de las variables y pulse el botón.')

# Se conserva el flujo de la app y se separa la clave del widget de la clave del valor editable.
ejemplo = pd.read_csv(BASE_DIR / 'data' / 'ejemplo_ventana.csv').tail(n).reset_index(drop=True)
editor_key = f'ventana_editor_{n}'
state_key = f'ventana_state_{n}'
if state_key not in st.session_state:
    st.session_state[state_key] = ejemplo.copy()
if st.sidebar.button('Restaurar valores originales'):
    st.session_state[state_key] = ejemplo.copy()

ventana = st.data_editor(st.session_state[state_key], key=editor_key, num_rows='fixed', width='stretch')
st.session_state[state_key] = ventana

# Información de la ventana y resultados del entrenamiento para la app.
st.sidebar.caption(f'Ventana del modelo: {n} horas')
resultados_path = BASE_DIR / 'resultados.csv'
if resultados_path.exists():
    try:
        df_res = pd.read_csv(resultados_path)
        fila = df_res[df_res['Ventana'] == n].copy()
        if not fila.empty:
            mejor = fila.sort_values('RMSE_val').iloc[0]
            st.sidebar.write(f"Arquitectura: {mejor['Modelo']}")
            st.sidebar.write(f"RMSE: {float(mejor['RMSE']):.2f}")
            st.sidebar.write(f"MAE: {float(mejor['MAE']):.2f}")
            st.sidebar.write(f"R²: {float(mejor['R2']):.3f}")
        else:
            st.sidebar.write('No hay resultados para esta ventana.')
    except Exception:
        st.sidebar.write('No se pudo leer resultados.csv; la app sigue funcionando.')
else:
    st.sidebar.write('No hay resultados.csv; la app sigue funcionando sin error.')

st.line_chart(ventana['demanda_mw'])

if st.button('Predecir demanda de la próxima hora'):
    errores = validar(ventana)
    if errores:
        for error in errores:
            st.error(error)
    else:
        x = agregar_ciclicas(ventana)[features]
        x_s = sx.transform(x).reshape(1, n, len(features)).astype('float32')
        pred_s = modelo.predict(x_s, verbose=0)
        pred = sy.inverse_transform(pred_s)[0, 0]
        cambio = pred - float(ventana['demanda_mw'].iloc[-1])

        col1, col2 = st.columns(2)
        col1.metric('Demanda predicha (MW)', f'{pred:.2f}')
        col2.metric('Cambio respecto a la última hora (MW)', f'{cambio:+.2f}')

        historica = ventana['demanda_mw'].astype(float).copy()
        chart_df = pd.DataFrame({
            'Demanda histórica': historica.tolist() + [np.nan],
            'Demanda con predicción': [*historica.iloc[:-1].tolist(), historica.iloc[-1], pred],
        })
        st.line_chart(chart_df)

        df_descarga = ventana.copy()
        df_descarga['prediccion_proxima_hora_mw'] = np.nan
        df_descarga.iloc[-1, df_descarga.columns.get_loc('prediccion_proxima_hora_mw')] = pred
        csv = df_descarga.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Descargar ventana con predicción',
            data=csv,
            file_name='ventana_prediccion.csv',
            mime='text/csv',
        )
