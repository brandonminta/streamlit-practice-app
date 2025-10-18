import streamlit as st
import pandas as pd
import numpy as np
import pickle
import patsy as pt
import matplotlib.pyplot as plt
from datetime import date

# === Load model and data
@st.cache_resource
def load_model():
    with open("modelo_neg_binomial.pkl", "rb") as f:
        modelo = pickle.load(f)
    df_poblacion = pd.read_csv("data/poblacion_provincias_ecuador_2022.csv")
    provincias = sorted(df_poblacion["provincia"].unique().tolist())
    return modelo, df_poblacion, provincias

modelo, df_poblacion, provincias_entrenadas = load_model()

# === Streamlit UI ===
st.set_page_config(page_title="ECU911 Predictor", layout="centered")
st.title("Predicción de emergencias ECU911")

# Sidebar inputs
st.sidebar.header("Parámetros de predicción")
provincia = st.sidebar.selectbox("Provincia", provincias_entrenadas)
start_date = st.sidebar.date_input("Fecha de inicio", date(2025, 1, 1))
end_date = st.sidebar.date_input("Fecha final", date(2025, 3, 31))

if start_date > end_date:
    st.error("La fecha inicial debe ser anterior a la final.")
    st.stop()

# === Prediction function (inline)
def predecir_incidentes(provincia, start_date, end_date, modelo, df_poblacion, provincias_entrenadas):
    fechas = pd.date_range(start=start_date, end=end_date, freq="D")
    df_futuro = pd.DataFrame({"fecha": fechas})
    pop = df_poblacion.loc[df_poblacion["provincia"] == provincia, "poblacion_2022"].values[0]
    df_futuro["provincia"] = provincia
    df_futuro["log_pop"] = np.log(pop)
    df_futuro["anio"] = df_futuro["fecha"].dt.year
    df_futuro["mes"] = df_futuro["fecha"].dt.month
    df_futuro["dia_semana"] = df_futuro["fecha"].dt.dayofweek
    df_futuro["es_fin_semana"] = df_futuro["dia_semana"].isin([5, 6]).astype(int)
    df_futuro["es_feriado"] = 0
    df_futuro["sin_mes"] = np.sin(2 * np.pi * df_futuro["mes"] / 12)
    df_futuro["cos_mes"] = np.cos(2 * np.pi * df_futuro["mes"] / 12)

    cat_dtype = pd.api.types.CategoricalDtype(categories=provincias_entrenadas)
    df_futuro["provincia"] = df_futuro["provincia"].astype(cat_dtype)

    rhs = "C(provincia) + es_fin_semana + C(dia_semana) + es_feriado + sin_mes + cos_mes + anio"
    X_fut = pt.dmatrix(rhs, df_futuro, return_type="dataframe")

    df_futuro["pred_nb"] = modelo.predict(X_fut, offset=df_futuro["log_pop"])
    total = df_futuro["pred_nb"].sum()
    return df_futuro, total

# === Prediction button
if st.sidebar.button("Predecir"):
    df_pred, total = predecir_incidentes(
        provincia, start_date, end_date, modelo, df_poblacion, provincias_entrenadas
    )

    st.success(f"**Total estimado de emergencias ({provincia})**: {total:,.0f}")
    st.write(f"Período: {start_date} → {end_date} ({len(df_pred)} días)")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_pred["fecha"], df_pred["pred_nb"], color="darkorange", lw=2)
    ax.set_title(f"Predicción diaria — {provincia}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Emergencias estimadas")
    st.pyplot(fig)

    with st.expander("Ver datos diarios"):
        st.dataframe(df_pred[["fecha", "pred_nb"]].rename(columns={"pred_nb": "Predicción"}))

else:
    st.info("Selecciona una provincia y un rango de fechas, luego presiona **Predecir**.")

