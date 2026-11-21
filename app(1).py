# -*- coding: utf-8 -*-
"""
EC1 - Modelamiento Predictivo de Datos
Web conjunta de las 4 dimensiones (Streamlit)

  1. CARRASCO   — Proyección de servicios (Holt optimizado + benchmarks)
  2. REYES      — Regresión lineal múltiple para reparto multimodal
  3. VILLANUEVA — Optimización de la cadena de abastecimiento (XGBoost, SS y ROP)
  4. HUAPAYA    — Integración con Power BI y R (Logística L1 + Random Forest)

Ejecución:  streamlit run app.py
Dataset:    global_supply_chain_risk_2026.csv (en la raíz o en data/),
            o cargarlo desde la barra lateral.
"""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBClassifier

st.set_page_config(
    page_title="EC1 — Modelamiento Predictivo de Datos",
    page_icon="🚚",
    layout="wide",
)

# =============================================================================
# CARGA DE DATOS COMPARTIDA
# =============================================================================
CANDIDATOS_CSV = [
    os.path.join("data", "global_supply_chain_risk_2026.csv"),
    "global_supply_chain_risk_2026.csv",
    "modelamiento.csv",
    os.path.join("/content", "global_supply_chain_risk_2026.csv"),
]


@st.cache_data(show_spinner=False)
def cargar_raw(archivo) -> pd.DataFrame:
    """Lee el CSV una sola vez. Cada dimensión trabaja sobre su propia copia."""
    if archivo is not None:
        return pd.read_csv(archivo)
    ruta = next((p for p in CANDIDATOS_CSV if os.path.exists(p)), None)
    if ruta is None:
        raise FileNotFoundError(
            "No se encontró el dataset. Coloca `global_supply_chain_risk_2026.csv` "
            "en la raíz o en `data/`, o cárgalo desde la barra lateral."
        )
    return pd.read_csv(ruta)


# =============================================================================
# 1. DIMENSIÓN CARRASCO — Proyección de servicios con Holt
# =============================================================================
COVARIABLES = [
    "Distancia_Promedio", "Indice_Combustible", "Riesgo_Geopolitico",
    "Fiabilidad_Carrier", "Lead_Time_Promedio", "Tonelaje_Total",
]

VARIANTES_HOLT = {
    "SES (nivel)": dict(trend=None, damped_trend=False),
    "Holt (tendencia)": dict(trend="add", damped_trend=False),
    "Holt amortiguado": dict(trend="add", damped_trend=True),
}


@st.cache_data(show_spinner=False)
def carrasco_agregar_datos(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df["Fecha_Dia"] = df["Date"].dt.date

    sem = df.set_index("Date").resample("W-MON").agg(
        Demanda_Servicios=("Shipment_ID", "count"),
        Dias_Activos=("Fecha_Dia", "nunique"),
        Tonelaje_Total=("Weight_MT", "sum"),
        Distancia_Promedio=("Distance_km", "mean"),
        Indice_Combustible=("Fuel_Price_Index", "mean"),
        Riesgo_Geopolitico=("Geopolitical_Risk_Score", "mean"),
        Fiabilidad_Carrier=("Carrier_Reliability_Score", "mean"),
        Lead_Time_Promedio=("Lead_Time_Days", "mean"),
        Tasa_Disrupcion=("Disruption_Occurred", "mean"),
    )
    sem = sem[sem["Dias_Activos"] == 7].drop(columns=["Dias_Activos"])
    sem = sem.asfreq("W-MON")
    sem["Demanda_Servicios"] = sem["Demanda_Servicios"].interpolate(limit=2)
    sem = sem.interpolate(limit=2).dropna()

    sem["Media_Movil_4"] = sem["Demanda_Servicios"].rolling(4).mean()
    return sem.dropna()


def _ajustar(historia, variante: str, manual=None):
    """Ajusta una variante Holt. manual = (alpha, beta, phi) o None (MLE)."""
    cfg = VARIANTES_HOLT[variante]
    mod = ExponentialSmoothing(
        np.asarray(historia, dtype=float),
        initialization_method="estimated", **cfg,
    )
    if manual is None:
        return mod.fit()
    alpha, beta, phi = manual
    kw = dict(smoothing_level=alpha, optimized=False)
    if cfg["trend"]:
        kw["smoothing_trend"] = beta
        if cfg["damped_trend"]:
            kw["damping_trend"] = phi
    return mod.fit(**kw)


def rolling_origin(serie: pd.Series, variante: str, inicio: int, manual=None):
    vals = serie.values.astype(float)
    preds = []
    for t in range(inicio, len(vals)):
        preds.append(float(_ajustar(vals[:t], variante, manual).forecast(1)[0]))
    return np.array(preds)


def benchmarks_rolling(serie: pd.Series, inicio: int) -> dict:
    v = serie.values.astype(float)
    idx = range(inicio, len(v))
    return {
        "Media histórica": np.array([v[:t].mean() for t in idx]),
        "Naive (última sem.)": np.array([v[t - 1] for t in idx]),
        "Media móvil 4 sem.": np.array([v[t - 4:t].mean() for t in idx]),
    }


def seleccionar_variante(serie: pd.Series, inicio: int, tolerancia=0.02):
    reales = serie.values[inicio:]
    maes, preds = {}, {}
    for nombre in VARIANTES_HOLT:
        preds[nombre] = rolling_origin(serie, nombre, inicio)
        maes[nombre] = mean_absolute_error(reales, preds[nombre])
    mejor = min(maes.values())
    for nombre in VARIANTES_HOLT:
        if maes[nombre] <= mejor * (1 + tolerancia):
            return nombre, maes, preds
    return min(maes, key=maes.get), maes, preds


def cobertura_intervalo(error_cv: np.ndarray, min_hist: int = 8) -> float:
    e = np.asarray(error_cv, dtype=float)
    dentro = [abs(e[i]) <= 1.96 * np.std(e[:i], ddof=1) for i in range(min_hist, len(e))]
    return 100.0 * float(np.mean(dentro)) if dentro else float("nan")


def backtest_acumulado(serie: pd.Series, variante: str, semanas: int,
                       n_test: int, manual=None):
    v = serie.values.astype(float)
    paso = max(2, semanas // 2)
    errs = []
    for t in range(len(v) - n_test, len(v) - semanas + 1, paso):
        f = float(np.sum(_ajustar(v[:t], variante, manual).forecast(semanas)))
        r = float(v[t:t + semanas].sum())
        errs.append(abs(f - r) / r * 100)
    return (float(np.mean(errs)) if errs else float("nan")), len(errs)


def proyectar(serie: pd.Series, variante: str, semanas: int,
              error_cv: np.ndarray, manual=None):
    modelo = _ajustar(serie.values, variante, manual)
    pron = np.asarray(modelo.forecast(semanas), dtype=float)

    alpha = float(modelo.params["smoothing_level"])
    h = np.arange(1, semanas + 1)
    sigma = float(np.std(error_cv, ddof=1))
    esc = np.sqrt(1 + (h - 1) * alpha ** 2)
    marg = 2.06 * sigma * esc

    fechas = pd.date_range(serie.index[-1] + pd.Timedelta(weeks=1),
                           periods=semanas, freq="W-MON")
    df = pd.DataFrame({
        "Fecha_Semana": fechas,
        "Proyeccion_Holt": np.round(pron, 1),
        "Limite_Inferior_95": np.round(np.maximum(pron - marg, 0), 1),
        "Limite_Superior_95": np.round(pron + marg, 1),
    })
    total = float(pron.sum())
    marg_tot = 2.06 * sigma * np.sqrt(semanas) * np.sqrt(1 + alpha ** 2 * (semanas - 1) / 2)
    return df, modelo, (total, max(total - marg_tot, 0), total + marg_tot)


def carrasco_datos_entrada(df: pd.DataFrame):
    st.subheader("1. Datos de entrada y reducción PCA")
    cols = ["Demanda_Servicios"] + COVARIABLES
    desc = df[cols].describe().T[["mean", "std", "min", "max"]]
    desc["moda"] = df[cols].mode().iloc[0]
    desc = desc[["mean", "moda", "std", "min", "max"]].round(2)
    desc.columns = ["Media", "Moda", "Desv. estándar", "Mínimo", "Máximo"]
    st.dataframe(desc, use_container_width=True)

    y = df["Demanda_Servicios"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Semanas completas", f"{len(df)}")
    c2.metric("Demanda promedio", f"{y.mean():.1f} desp./sem")
    c3.metric("Autocorrelación lag-1", f"{y.autocorr(1):.2f}",
              help="Cerca de 0 = poca memoria temporal: Holt tendrá poco que explotar.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Demanda vs. media móvil (4 sem.)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df.index, y, color="#1B4F72", label="Demanda real")
        ax.plot(df.index, df["Media_Movil_4"], color="#D35400", ls="--", label="Media móvil 4")
        ax.axhline(y.mean(), color="gray", ls=":", label="Media global")
        ax.legend(fontsize=8)
        ax.grid(True, ls=":", alpha=.6)
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        st.markdown("**PCA sobre covariables**")
        Xs = StandardScaler().fit_transform(df[COVARIABLES])
        pca = PCA(n_components=2)
        Z = pca.fit_transform(Xs)
        v = pca.explained_variance_ratio_
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", alpha=.85)
        ax.set_xlabel(f"PC1 ({v[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({v[1]*100:.1f}%)")
        ax.set_title(f"Varianza acumulada = {(v[0]+v[1])*100:.1f}%")
        plt.colorbar(sc, ax=ax, label="Demanda")
        ax.grid(True, ls=":", alpha=.6)
        st.pyplot(fig)
        plt.close(fig)


def _metricas(real, pred):
    mae = mean_absolute_error(real, pred)
    rmse = np.sqrt(mean_squared_error(real, pred))
    mape = np.mean(np.abs(real - pred) / real) * 100
    return mae, rmse, mape


def carrasco_modelo_y_proyeccion(df: pd.DataFrame):
    st.subheader("2. Selección de modelo y proyección con Holt")

    c1, c2, c3 = st.columns(3)
    n_test = c1.slider("Semanas de validación (rolling)", 16, 40, 26, step=2, key="c_ntest")
    semanas = c2.slider("Semanas a proyectar", 4, 16, 8, step=2, key="c_sem")
    modo = c3.radio("Parámetros Holt", ["Automático (MLE + CV)", "Manual"], key="c_modo")

    manual, variante_manual = None, "Holt amortiguado"
    if modo == "Manual":
        m1, m2, m3, m4 = st.columns(4)
        variante_manual = m1.selectbox("Variante", list(VARIANTES_HOLT), index=2, key="c_var")
        a = m2.slider("Alpha", 0.05, 0.90, 0.20, 0.05, key="c_a")
        b = m3.slider("Beta", 0.01, 0.30, 0.05, 0.01, key="c_b")
        p = m4.slider("Phi (amortig.)", 0.80, 0.99, 0.90, 0.01, key="c_p")
        manual = (a, b, p)

    y = df["Demanda_Servicios"]
    inicio = len(y) - n_test
    reales = y.values[inicio:]
    fechas_test = y.index[inicio:]

    with st.spinner("Validando modelos (origen móvil)…"):
        if manual is None:
            variante, maes_cv, preds_cv = seleccionar_variante(y, inicio)
            pred_holt = preds_cv[variante]
            etiqueta = f"Holt — {variante} [Seleccionado]"
        else:
            variante = variante_manual
            pred_holt = rolling_origin(y, variante, inicio, manual)
            etiqueta = f"Holt — {variante} (manual) [Seleccionado]"
        extra = {}
        if manual is None:
            for k, v in preds_cv.items():
                if k != variante:
                    extra[f"Holt — {k}"] = v
        bench = benchmarks_rolling(y, inicio)

        # Ridge / SVR sin fuga: solo información disponible en t-1
        X = pd.DataFrame(index=df.index)
        X["Lag_1"] = y.shift(1)
        X["Lag_2"] = y.shift(2)
        X["MM4_prev"] = y.shift(1).rolling(4).mean()
        for c in COVARIABLES:
            X[c + "_prev"] = df[c].shift(1)
        X["Mes"] = df.index.month
        ok = X.dropna().index
        corte = X.index.get_loc(y.index[inicio])
        tr_idx = [i for i in ok if X.index.get_loc(i) < corte]
        te_idx = y.index[inicio:]
        sc = StandardScaler().fit(X.loc[tr_idx])
        Xtr, Xte = sc.transform(X.loc[tr_idx]), sc.transform(X.loc[te_idx])
        ytr = y.loc[tr_idx]
        pred_ridge = Ridge(alpha=10.0).fit(Xtr, ytr).predict(Xte)
        pred_svr = SVR(kernel="rbf", C=10.0, epsilon=0.5).fit(Xtr, ytr).predict(Xte)

    todos = {etiqueta: pred_holt, **extra, **bench,
             "Ridge (L2, sin fuga)": pred_ridge, "SVR (RBF, sin fuga)": pred_svr}
    filas = []
    for nombre, pred in todos.items():
        mae, rmse, mape = _metricas(reales, pred)
        sse_ref = np.sum((reales - bench["Media histórica"]) ** 2)
        r2_ref = 1 - np.sum((reales - pred) ** 2) / sse_ref
        filas.append({"Modelo": nombre, "MAE": round(mae, 2),
                      "RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2),
                      "R² vs media hist.": round(r2_ref, 3)})
    tabla = pd.DataFrame(filas).set_index("Modelo").sort_values("MAE")
    st.dataframe(tabla, use_container_width=True)

    mae_holt = tabla.loc[etiqueta, "MAE"]
    mae_ref = min(tabla.loc[k, "MAE"] for k in bench)
    mejora = (mae_ref - mae_holt) / mae_ref * 100

    err_cv = reales - pred_holt
    df_proy, modelo, (tot, tot_lo, tot_hi) = proyectar(y, variante, semanas, err_cv, manual)
    cob = cobertura_intervalo(err_cv)
    mape_acum, n_bloques = backtest_acumulado(y, variante, semanas, n_test, manual)

    st.markdown("**Confiabilidad de la proyección**")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Promedio proyectado", f"{df_proy['Proyeccion_Holt'].mean():.1f} /sem")
    k2.metric(f"Total {semanas} semanas", f"{tot:.0f}",
              help=f"Intervalo 95%: {tot_lo:.0f} – {tot_hi:.0f}")
    k3.metric("Error al proyectar el total",
              "n/d" if np.isnan(mape_acum) else f"{mape_acum:.1f}%",
              help=f"Backtest sobre {n_bloques} bloques de {semanas} semanas. "
                   "El ruido semanal se compensa al sumar.")
    k4.metric("Cobertura intervalo 95%",
              "n/d" if np.isnan(cob) else f"{cob:.0f}%",
              help="% de semanas reales que cayeron dentro del intervalo en validación.")
    st.caption(f"Total proyectado a {semanas} semanas: **{tot:.0f}** despachos "
               f"(intervalo 95%: {tot_lo:.0f} – {tot_hi:.0f}).")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Validación fuera de muestra**")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(fechas_test, reales, "k-o", ms=4, lw=2, label="Real")
        ax.plot(fechas_test, pred_holt, color="#E67E22", lw=2.2, label="Holt")
        ax.plot(fechas_test, bench["Media histórica"], color="gray", ls=":", label="Media histórica")
        ax.plot(fechas_test, pred_ridge, color="#2980B9", ls="--", alpha=.8, label="Ridge")
        ax.legend(fontsize=8)
        ax.grid(True, ls=":", alpha=.6)
        plt.xticks(rotation=25)
        st.pyplot(fig)
        plt.close(fig)
    with g2:
        st.markdown(f"**Proyección (+{semanas} semanas)**")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        h = y.iloc[-24:]
        ax.plot(h.index, h.values, color="#1B4F72", marker="o", ms=3, label="Histórico")
        ax.plot(df_proy["Fecha_Semana"], df_proy["Proyeccion_Holt"], color="#D35400",
                lw=2.4, marker="s", ms=4, label="Proyección Holt")
        ax.fill_between(df_proy["Fecha_Semana"], df_proy["Limite_Inferior_95"],
                        df_proy["Limite_Superior_95"], color="#F5B041", alpha=.25,
                        label="Intervalo 95%")
        ax.axhline(y.mean(), color="gray", ls=":", label="Media histórica")
        ax.axvline(h.index[-1], color="gray", ls="--", lw=1)
        ax.legend(fontsize=8)
        ax.grid(True, ls=":", alpha=.6)
        plt.xticks(rotation=25)
        st.pyplot(fig)
        plt.close(fig)

    p = modelo.params
    st.caption(
        f"Parámetros usados: α={p['smoothing_level']:.3f}"
        + (f", β={p['smoothing_trend']:.3f}" if variante != "SES (nivel)" else "")
        + (f", φ={p['damping_trend']:.3f}" if variante == "Holt amortiguado" else "")
    )

    texto = (f"**Modelo:** {etiqueta}. MAE de validación = {mae_holt} desp./sem "
             f"(MAPE {tabla.loc[etiqueta, 'MAPE (%)']}%). "
             f"Proyección: **~{df_proy['Proyeccion_Holt'].mean():.1f} servicios/semana**. ")
    if mejora > 3:
        st.success(texto + f"Mejora al mejor benchmark en {mejora:.1f}%.")
    else:
        st.info(texto + "Su ventaja sobre el mejor benchmark simple es "
                f"{mejora:.1f}%: la serie casi no tiene estructura temporal, "
                "así que la proyección equivale a un nivel estable.")
    return df_proy, modelo, variante


def carrasco_integracion(df_proy, modelo, variante):
    st.subheader("3. Tabla de proyección y exportación")
    st.dataframe(df_proy, use_container_width=True, hide_index=True)

    p = modelo.params
    a = round(float(p["smoothing_level"]), 4)
    b = round(float(p.get("smoothing_trend", 0) or 0), 4)
    phi = round(float(p.get("damping_trend", 1) or 1), 4)
    h = len(df_proy)
    if variante == "SES (nivel)":
        llamada = f"ses(serie_semanal, h={h}, alpha={a}, initial='optimal')"
    else:
        amort = "TRUE" if variante == "Holt amortiguado" else "FALSE"
        llamada = (f"holt(serie_semanal, h={h}, damped={amort}, alpha={a}, "
                   f"beta={b}" + (f", phi={phi}" if amort == "TRUE" else "") + ")")
    codigo_r = f"""# Dimensión CARRASCO — {variante}
library(forecast)
datos <- read.csv("global_supply_chain_risk_2026.csv")
datos$Date <- as.Date(datos$Date)
datos$semana <- cut(datos$Date, "week", start.on.monday = TRUE)
conteo <- as.data.frame(table(datos$semana))
# Excluir semanas incompletas (primera/última) antes de continuar
serie_semanal <- ts(conteo$Freq, frequency = 52)
modelo <- {llamada}
print(summary(modelo))
plot(modelo)
"""
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Proyección CSV (Power BI)",
                       df_proy.to_csv(index=False).encode("utf-8"),
                       file_name="carrasco_proyeccion_futura_holt.csv", key="c_dl_csv")
    c2.download_button("⬇️ Script R", codigo_r.encode("utf-8"),
                       file_name="carrasco_holt.R", key="c_dl_r")


def render_carrasco(raw: pd.DataFrame):
    st.title("📈 Dimensión CARRASCO — Proyección de Servicios con Holt")
    df = carrasco_agregar_datos(raw)
    carrasco_datos_entrada(df)
    st.divider()
    df_proy, modelo, variante = carrasco_modelo_y_proyeccion(df)
    st.divider()
    carrasco_integracion(df_proy, modelo, variante)


# =============================================================================
# 2. DIMENSIÓN REYES — Regresión lineal múltiple para reparto multimodal
# =============================================================================
R_TARGET = "Lead_Time_Days"
R_NUMERIC = [
    "Distance_km", "Weight_MT", "Fuel_Price_Index",
    "Geopolitical_Risk_Score", "Carrier_Reliability_Score",
]
R_CATEGORICAL = [
    "Transport_Mode", "Weather_Condition", "Product_Category",
    "Origin_Port", "Destination_Port",
]
R_FEATURES = R_NUMERIC + R_CATEGORICAL
R_REQUIRED = R_FEATURES + [R_TARGET, "Date"]
R_TEST_SIZE = 0.25
R_SEED = 42

R_CODE = r'''# REGRESION LINEAL MULTIPLE PARA PREDECIR LEAD TIME — POWER BI (R VISUAL)
# IMPORTANTE: Power BI crea automaticamente el data.frame llamado `dataset`.
# En Values agrega estas columnas EXACTAS:
# Shipment_ID (solo identifica cada envio y evita que Power BI agrupe duplicados),
# Lead_Time_Days, Distance_km, Weight_MT, Fuel_Price_Index,
# Geopolitical_Risk_Score, Carrier_Reliability_Score, Transport_Mode,
# Weather_Condition, Product_Category, Origin_Port y Destination_Port.
# Shipment_ID NO se usa como predictor del modelo.
# No se requieren paquetes externos: se utilizan funciones base de R.

# 1. Validar que Power BI haya enviado las columnas necesarias
columnas_requeridas <- c(
  "Shipment_ID", "Lead_Time_Days", "Distance_km", "Weight_MT",
  "Fuel_Price_Index", "Geopolitical_Risk_Score",
  "Carrier_Reliability_Score", "Transport_Mode", "Weather_Condition",
  "Product_Category", "Origin_Port", "Destination_Port"
)

if (!exists("dataset") || !is.data.frame(dataset)) {
  stop("Power BI no entrego el objeto 'dataset'. Agrega las columnas al area Values del R Visual.")
}

faltantes <- setdiff(columnas_requeridas, names(dataset))
if (length(faltantes) > 0) {
  stop(paste("Faltan columnas en Values:", paste(faltantes, collapse = ", ")))
}

datos <- as.data.frame(dataset[, columnas_requeridas, drop = FALSE])

# 2. Preparar tipos de datos
variables_numericas <- c(
  "Lead_Time_Days", "Distance_km", "Weight_MT", "Fuel_Price_Index",
  "Geopolitical_Risk_Score", "Carrier_Reliability_Score"
)
variables_categoricas <- c(
  "Transport_Mode", "Weather_Condition", "Product_Category",
  "Origin_Port", "Destination_Port"
)

datos$Shipment_ID <- trimws(as.character(datos$Shipment_ID))
for (v in variables_numericas) {
  datos[[v]] <- suppressWarnings(as.numeric(as.character(datos[[v]])))
}
for (v in variables_categoricas) {
  datos[[v]] <- trimws(as.character(datos[[v]]))
  datos[[v]][datos[[v]] == ""] <- NA_character_
}

datos <- datos[complete.cases(datos), , drop = FALSE]
datos <- datos[!is.na(datos$Shipment_ID) & datos$Shipment_ID != "", , drop = FALSE]
# Ordenar por identificador hace reproducible la particion aun si Power BI cambia el orden de filas.
datos <- datos[order(datos$Shipment_ID), , drop = FALSE]

if (nrow(datos) < 30) {
  stop(paste("Solo hay", nrow(datos), "envios completos. Se requieren al menos 30 para entrenar y evaluar el modelo."))
}

# 3. Dividir aleatoriamente: 75% entrenamiento y 25% prueba
set.seed(42)
n <- nrow(datos)
indices_train <- sample(seq_len(n), size = floor(0.75 * n))
train <- datos[indices_train, , drop = FALSE]
test <- datos[-indices_train, , drop = FALSE]

# 4. Preparar variables categoricas sin niveles nuevos en el conjunto de prueba
# Si aparece una categoria que no estuvo en entrenamiento, se asigna a la categoria
# mas frecuente del entrenamiento para evitar errores de predict.lm().
for (v in variables_categoricas) {
  valores_train <- trimws(as.character(train[[v]]))
  valores_test <- trimws(as.character(test[[v]]))
  niveles_train <- sort(unique(valores_train))

  if (length(niveles_train) == 0) {
    next
  }

  frecuencia <- sort(table(valores_train), decreasing = TRUE)
  categoria_referencia <- names(frecuencia)[1]
  valores_test[is.na(valores_test) | !(valores_test %in% niveles_train)] <- categoria_referencia

  train[[v]] <- factor(valores_train, levels = niveles_train)
  test[[v]] <- factor(valores_test, levels = niveles_train)
}

# 5. Estimar la regresion lineal multiple (minimos cuadrados ordinarios)
# Se excluyen automaticamente predictores constantes en el conjunto de entrenamiento.
num_modelo <- c(
  "Distance_km", "Weight_MT", "Fuel_Price_Index",
  "Geopolitical_Risk_Score", "Carrier_Reliability_Score"
)
num_modelo <- num_modelo[vapply(train[num_modelo], function(x) {
  is.finite(stats::sd(x)) && stats::sd(x) > 0
}, logical(1))]

cat_modelo <- variables_categoricas[vapply(train[variables_categoricas], function(x) {
  nlevels(droplevels(x)) > 1
}, logical(1))]

predictores <- c(num_modelo, cat_modelo)
if (length(predictores) == 0) {
  stop("No hay variables predictoras con variacion suficiente para estimar la regresion.")
}

formula_modelo <- stats::reformulate(predictores, response = "Lead_Time_Days")
modelo <- stats::lm(formula_modelo, data = train)

# 6. Predecir el Lead Time del 25% reservado para prueba
predicciones <- as.numeric(stats::predict(modelo, newdata = test))
lead_time_real <- test$Lead_Time_Days
residuos <- lead_time_real - predicciones

validos <- is.finite(predicciones) & is.finite(lead_time_real)
predicciones <- predicciones[validos]
lead_time_real <- lead_time_real[validos]
residuos <- residuos[validos]
modalidad <- as.character(test$Transport_Mode[validos])

if (length(predicciones) < 2) {
  stop("El modelo no produjo suficientes predicciones validas para graficar.")
}

# 7. Indicadores de evaluacion
RMSE <- sqrt(mean(residuos^2))
MAE <- mean(abs(residuos))
SCT <- sum((lead_time_real - mean(lead_time_real))^2)
R2 <- if (SCT > 0) 1 - sum(residuos^2) / SCT else NA_real_

# 8. Tabla resumen: promedios reales y predichos por modalidad observada
resultados <- data.frame(
  Lead_Time_Real = lead_time_real,
  Lead_Time_Predicho = predicciones,
  Residuo = residuos,
  Transport_Mode = modalidad
)
resumen <- aggregate(
  cbind(Lead_Time_Real, Lead_Time_Predicho) ~ Transport_Mode,
  data = resultados,
  FUN = mean
)
resumen <- resumen[order(resumen$Transport_Mode), , drop = FALSE]

# 9. Dibujar los tres graficos en un solo R Visual de Power BI
# Fila superior: dispersion real vs. predicho y residuos.
# Fila inferior: grafico de barras a lo ancho para facilitar su lectura.
graphics::layout(matrix(c(1, 2, 3, 3), nrow = 2, byrow = TRUE), heights = c(1, 1))
par(mar = c(4.2, 4.2, 3.0, 1.0), oma = c(0.5, 0.5, 2.2, 0.5), cex = 0.85)

# Grafico 1: Lead Time real vs. predicho
limites <- range(c(lead_time_real, predicciones), finite = TRUE)
if (diff(limites) == 0) limites <- limites + c(-1, 1)
plot(
  lead_time_real, predicciones,
  pch = 16, cex = 0.55, col = grDevices::adjustcolor("#0072B2", alpha.f = 0.45),
  xlim = limites, ylim = limites,
  xlab = "Lead Time real (dias)", ylab = "Lead Time predicho (dias)",
  main = "Real vs. predicho"
)
abline(a = 0, b = 1, lty = 2, col = "#0072B2", lwd = 1.5)

# Grafico 2: residuos vs. predicciones
plot(
  predicciones, residuos,
  pch = 16, cex = 0.55, col = grDevices::adjustcolor("#0072B2", alpha.f = 0.45),
  xlab = "Lead Time predicho (dias)", ylab = "Residuo (real - predicho)",
  main = "Residuos vs. predicciones"
)
abline(h = 0, lty = 2, col = "#0072B2", lwd = 1.5)

# Grafico 3: promedio real vs. predicho por modo de transporte
alturas <- rbind(resumen$Lead_Time_Real, resumen$Lead_Time_Predicho)
barplot(
  alturas, beside = TRUE,
  names.arg = resumen$Transport_Mode,
  col = c("#0072B2", "#E66C1C"),
  ylab = "Dias", xlab = "Modo de transporte",
  main = "Promedio real vs. predicho",
  cex.names = 0.85, ylim = c(0, max(alturas, na.rm = TRUE) * 1.18)
)
legend(
  "topleft", legend = c("Real", "Predicho"),
  fill = c("#0072B2", "#E66C1C"), bty = "n", cex = 0.85
)

# Metricas visibles en el encabezado del visual
mtext(
  paste0("Regresion lineal multiple | RMSE = ", round(RMSE, 2),
         " dias | MAE = ", round(MAE, 2), " dias | R2 = ", round(R2, 3)),
  side = 3, outer = TRUE, line = 0.5, cex = 0.95, font = 2
)
'''


def mode_value(series: pd.Series):
    values = series.mode(dropna=True)
    return values.iloc[0] if not values.empty else np.nan


def reyes_encoder() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [("num", StandardScaler(), R_NUMERIC), ("cat", encoder, R_CATEGORICAL)],
        remainder="drop",
    )


def reyes_preparar(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in R_REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError("Faltan columnas requeridas: " + ", ".join(missing))
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.drop_duplicates().reset_index(drop=True)
    return df.dropna(subset=R_REQUIRED).reset_index(drop=True)


@st.cache_resource(show_spinner="Entrenando regresión lineal…")
def reyes_fit_linear(df: pd.DataFrame):
    """Entrena regresión lineal múltiple (OLS) para predecir Lead Time."""
    X = df[R_FEATURES].copy()
    y = df[R_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=R_TEST_SIZE, random_state=R_SEED
    )
    pipe = Pipeline([
        ("pre", reyes_encoder()),
        ("model", LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))
    return pipe, X_test, y_test, pred, rmse, mae, r2


def reyes_coef_table(pipe: Pipeline) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    names = pipe.named_steps["pre"].get_feature_names_out()
    table = pd.DataFrame({"Variable": names, "Coeficiente": np.asarray(model.coef_).ravel()})
    table["Retenida"] = table["Coeficiente"].abs() > 1e-10
    return table.sort_values("Coeficiente", key=np.abs, ascending=False).reset_index(drop=True)


def reyes_pca(df: pd.DataFrame):
    Z = StandardScaler().fit_transform(df[R_NUMERIC])
    pca = PCA()
    scores = pca.fit_transform(Z)
    explained = pca.explained_variance_ratio_
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    tabla = pd.DataFrame({
        "Variable": R_NUMERIC,
        "Carga PC1": loadings[:, 0],
        "Carga PC2": loadings[:, 1],
    })
    return scores, explained, tabla


def reyes_plot_pca(scores, explained):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(scores[:, 0], scores[:, 1], alpha=0.28, s=15)
    ax.set_title(f"PCA — PC1 ({explained[0] * 100:.1f}%) vs PC2 ({explained[1] * 100:.1f}%)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def reyes_plot_mode_box(df: pd.DataFrame):
    modes = sorted(df["Transport_Mode"].unique())
    values = [df.loc[df["Transport_Mode"] == m, R_TARGET] for m in modes]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.boxplot(values, showfliers=False)
    ax.set_xticks(range(1, len(modes) + 1))
    ax.set_xticklabels(modes)
    ax.set_title("Lead Time por modo de transporte")
    ax.set_xlabel("Modo de transporte")
    ax.set_ylabel("Lead Time (días)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def reyes_plot_distance(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(df["Distance_km"], df[R_TARGET], alpha=0.25, s=12)
    ax.set_title("Distancia vs. Lead Time")
    ax.set_xlabel("Distancia (km)")
    ax.set_ylabel("Lead Time (días)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def reyes_plot_real_vs_pred(y_true, pred):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(y_true, pred, alpha=0.45, s=16)
    low = min(float(np.min(y_true)), float(np.min(pred)))
    high = max(float(np.max(y_true)), float(np.max(pred)))
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.2)
    ax.set_title("Regresión lineal — Lead Time real vs. predicho")
    ax.set_xlabel("Lead Time real (días)")
    ax.set_ylabel("Lead Time predicho (días)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def reyes_plot_residuals(y_true, pred):
    residuals = np.asarray(y_true) - np.asarray(pred)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(pred, residuals, alpha=0.35, s=15)
    ax.axhline(0, linestyle="--", linewidth=1.2)
    ax.set_title("Regresión lineal — Residuos vs. predicciones")
    ax.set_xlabel("Lead Time predicho (días)")
    ax.set_ylabel("Residuo (real - predicho)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def reyes_summary_by_mode(X_test: pd.DataFrame, y_test, pred_test) -> pd.DataFrame:
    """Resume el Lead Time real y predicho por modo, con los envíos del conjunto de prueba."""
    comparison = pd.DataFrame({
        "Transport_Mode": X_test["Transport_Mode"].to_numpy(),
        "Lead_Time_Real": np.asarray(y_test),
        "Lead_Time_Predicho": np.asarray(pred_test),
    })
    return (
        comparison.groupby("Transport_Mode", as_index=False)
        .agg(
            Lead_Time_Real=("Lead_Time_Real", "mean"),
            Lead_Time_Predicho=("Lead_Time_Predicho", "mean"),
            Envios=("Lead_Time_Real", "size"),
        )
        .sort_values("Transport_Mode")
        .reset_index(drop=True)
    )


def render_reyes(raw: pd.DataFrame):
    try:
        df = reyes_preparar(raw)
        linear_pipe, X_test, y_test, pred_test, rmse, mae, r2 = reyes_fit_linear(df)
    except Exception as exc:
        st.error(str(exc))
        return

    coef_table = reyes_coef_table(linear_pipe)

    st.title("🚚 Dimensión REYES — Regresión para reparto multimodal")
    st.caption("Regresión lineal múltiple para estimar el Lead Time de cada envío")

    with st.expander("1. Problema, pregunta y objetivo", expanded=True):
        st.markdown(
            "**Problema.** La gestión del reparto internacional debe considerar simultáneamente "
            "distancia, peso, condiciones económicas, riesgo geopolítico, fiabilidad del transportista, "
            "clima y modo de transporte. Estas características pueden generar variabilidad en el tiempo "
            "de entrega, dificultando la planificación del reparto."
        )
        st.markdown(
            "**Pregunta de investigación.** ¿En qué medida las características logísticas, económicas, "
            "ambientales y el modo de transporte permiten predecir el `Lead_Time_Days` de un envío y "
            "apoyar la selección de un modo de transporte con menor tiempo estimado?"
        )
        st.markdown(
            "**Objetivo.** Desarrollar y evaluar un modelo de regresión múltiple para predecir el "
            "`Lead_Time_Days` del proceso de reparto internacional, incorporando el modo de transporte "
            "y las principales características del envío, y utilizar la predicción para comparar escenarios multimodales."
        )
        st.markdown("**Indicadores del dataset:**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lead Time medio", f"{df[R_TARGET].mean():.2f} días")
        c2.metric("Desv. estándar", f"{df[R_TARGET].std():.2f} días")
        c3.metric("Mínimo", f"{df[R_TARGET].min():.2f} días")
        c4.metric("Máximo", f"{df[R_TARGET].max():.2f} días")

    with st.expander("2. Datos de entrada, análisis descriptivo y gráficos"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros", f"{len(df):,}")
        c2.metric("Variables", f"{df.shape[1]}")
        c3.metric("Variables analizadas", "11")

        st.markdown("**Variable respuesta (Y):** `Lead_Time_Days`")
        st.markdown(
            "**Variables predictoras (X):** `Distance_km`, `Weight_MT`, `Fuel_Price_Index`, "
            "`Geopolitical_Risk_Score`, `Carrier_Reliability_Score`, `Transport_Mode`, "
            "`Weather_Condition`, `Product_Category`, `Origin_Port` y `Destination_Port`. "
            "Las variables categóricas se incorporan mediante indicadores (dummies)."
        )
        st.caption(
            "El dataset tiene 14 columnas. `Shipment_ID` es un identificador, `Date` es referencia temporal "
            "y `Disruption_Occurred` pertenece a otra dimensión; por eso no se utilizan como predictores aquí."
        )

        st.markdown("### Estadísticos descriptivos — cuantitativas")
        desc_rows = []
        for col in R_NUMERIC + [R_TARGET]:
            s = df[col]
            desc_rows.append({
                "Variable": col, "Media": s.mean(), "Moda": mode_value(s),
                "Desv. estándar": s.std(ddof=1), "Mínimo": s.min(), "Máximo": s.max(),
            })
        st.dataframe(pd.DataFrame(desc_rows).round(3), use_container_width=True)

        st.markdown("### Resumen de variables categóricas")
        cat_rows = []
        for col in R_CATEGORICAL:
            m = mode_value(df[col])
            cat_rows.append({
                "Variable": col, "N.º categorías": int(df[col].nunique()),
                "Moda": m, "Frecuencia moda": int((df[col] == m).sum()),
            })
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True)

        p1, p2 = st.columns(2)
        with p1:
            fig, ax = plt.subplots(figsize=(7.5, 4.6))
            ax.hist(df[R_TARGET], bins=35)
            ax.set_title("Distribución del Lead Time")
            ax.set_xlabel("Lead Time (días)")
            ax.set_ylabel("Frecuencia")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        with p2:
            st.pyplot(reyes_plot_mode_box(df), clear_figure=True)

        st.pyplot(reyes_plot_distance(df), clear_figure=True)

    with st.expander("3. Reducción de dimensiones — PCA"):
        scores, explained, loading_table = reyes_pca(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("PC1", f"{explained[0] * 100:.1f}%")
        c2.metric("PC2", f"{explained[1] * 100:.1f}%")
        c3.metric("PC1 + PC2", f"{(explained[0] + explained[1]) * 100:.1f}%")
        st.pyplot(reyes_plot_pca(scores, explained), clear_figure=True)
        st.markdown("### Cargas de PC1 y PC2")
        st.dataframe(loading_table.round(4), use_container_width=True)
        st.caption(
            "PC1 es una nueva dimensión construida a partir de las variables cuantitativas originales. "
            "Las cargas nos permiten identificar qué variables tienen mayor peso en la construcción de ese componente."
        )

    with st.expander("4. Características del modelo de predicción"):
        st.markdown("**Tipo:** modelo supervisado de regresión lineal múltiple.")
        st.markdown("**Modelo:** regresión lineal múltiple clásica (OLS).")
        st.markdown("**Técnica:** mínimos cuadrados ordinarios (OLS).")
        st.markdown(
            "**Predictores:** variables numéricas y categóricas del envío. "
            "Las categorías se representan mediante variables indicadoras (dummies)."
        )
        st.markdown("**¿Cómo funciona?** La regresión lineal estima los coeficientes que minimizan la suma de cuadrados de los residuos:")
        st.latex(r"\min_{\beta} \sum_i (y_i-\hat y_i)^2")
        st.caption("Los coeficientes estimados permiten interpretar la relación de los predictores con el Lead Time, manteniendo los demás constantes.")
        st.markdown(
            "**Método:** mínimos cuadrados ordinarios (OLS), sin regularización. "
            "El modelo estima los coeficientes que mejor ajustan los datos de entrenamiento."
        )
        st.markdown(
            "**Evaluación:** RMSE mide el error de predicción en días y R² indica la proporción de variabilidad "
            "del Lead Time explicada por el modelo."
        )
        st.caption("El modelo se estima mediante mínimos cuadrados ordinarios (OLS), sin penalización Lasso.")

    with st.expander("5. Regresión lineal para predecir el Lead Time", expanded=True):
        st.markdown("### 5.1. Ejecución del modelo")
        st.markdown(
            "Se desarrolla una **regresión lineal múltiple** para predecir el tiempo de entrega "
            "(`Lead_Time_Days`). El 75 % de los registros se utiliza para entrenar el modelo y el "
            "25 % restante para evaluar las predicciones. Se consideran variables logísticas y el "
            "modo de transporte; las variables categóricas se codifican como indicadores (dummies)."
        )
        st.markdown(
            "El modelo aprende las relaciones entre las características del envío y el Lead Time. "
            "Después estima el tiempo de entrega para registros que no fueron usados en el entrenamiento. "
            "Se evalúa mediante RMSE, MAE y R²."
        )
        st.markdown("### 5.2. Bosquejo del código en R")
        st.code(R_CODE, language="r")
        st.markdown(
            "**Para ejecutarlo en Power BI:** inserta un visual de R y agrega a `Values` las 12 columnas "
            "indicadas al inicio del código. Incluye `Shipment_ID` para conservar la identificación de "
            "cada envío; no se utiliza como predictor. Mantén las columnas numéricas en `No resumir`. "
            "El script utiliza el objeto `dataset` que Power BI crea automáticamente y genera los tres gráficos."
        )
        st.caption(
            "El script está escrito con funciones base de R; no requiere instalar paquetes adicionales. "
            "El dashboard de Streamlit ejecuta la misma metodología OLS en Python."
        )
        st.markdown("### 5.3. Resultados e interpretación")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse:.2f} días")
        c2.metric("MAE", f"{mae:.2f} días")
        c3.metric("R²", f"{r2:.3f}")
        st.markdown(
            f"En el conjunto de prueba, el RMSE es **{rmse:.2f} días** y el MAE es **{mae:.2f} días**. "
            f"El MAE representa el error absoluto promedio de predicción. El **R² de {r2:.3f}** indica que "
            f"el modelo explica aproximadamente **{r2 * 100:.1f}%** de la variabilidad observada del Lead Time. "
            "Las métricas evalúan predicción y no demuestran una relación causal."
        )
        p1, p2 = st.columns(2)
        with p1:
            st.pyplot(reyes_plot_real_vs_pred(y_test, pred_test), clear_figure=True)
            st.caption("Los puntos cercanos a la diagonal indican predicciones próximas al Lead Time real.")
        with p2:
            st.pyplot(reyes_plot_residuals(y_test, pred_test), clear_figure=True)
            st.caption("El residuo es la diferencia entre el valor real y el predicho; cerca de cero indica menor error.")

        st.markdown("**Coeficientes estimados**")
        st.dataframe(coef_table.round(4), use_container_width=True)
        st.caption(
            "Los coeficientes muestran la dirección de la asociación estimada, manteniendo las demás variables constantes. "
            "Las variables numéricas están estandarizadas en el preprocesamiento; las categorías se comparan con su referencia. "
            "Los coeficientes no representan por sí solos pruebas de significancia."
        )

        st.markdown("**Resumen por modalidad de transporte**")
        mode_summary = reyes_summary_by_mode(X_test, y_test, pred_test)
        st.dataframe(mode_summary.round(3), use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 4.6))
        positions = np.arange(len(mode_summary))
        width = 0.36
        ax.bar(positions - width / 2, mode_summary["Lead_Time_Real"], width, label="Real")
        ax.bar(positions + width / 2, mode_summary["Lead_Time_Predicho"], width, label="Predicho")
        ax.set_xticks(positions)
        ax.set_xticklabels(mode_summary["Transport_Mode"])
        ax.set_title("Lead Time promedio real vs. predicho por modo")
        ax.set_xlabel("Modo de transporte")
        ax.set_ylabel("Días")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption(
            "La tabla y el gráfico agrupan los envíos de prueba por su modo observado. "
            "No representan una simulación causal de cambiar la modalidad de un mismo envío."
        )


# =============================================================================
# 3. DIMENSIÓN VILLANUEVA — Optimización de la cadena (XGBoost, SS y ROP)
# =============================================================================
V_COLS_EXCLUIDAS = ["Shipment_ID", "Date", "Origin_Port", "Destination_Port", "Disruption_Occurred"]
V_CAT_COLS = ["Transport_Mode", "Product_Category", "Weather_Condition"]
V_FECHA_CORTE = pd.Timestamp("2025-01-01")


def _nuevo_xgb(spw):
    return XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=5, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=spw, random_state=42, eval_metric="logloss",
    )


@st.cache_resource(show_spinner="Entrenando XGBoost…")
def villanueva_train_model(df_input):
    df_encoded = pd.get_dummies(df_input, columns=V_CAT_COLS, drop_first=True)
    features = [c for c in df_encoded.columns if c not in V_COLS_EXCLUIDAS]
    X = df_encoded[features]
    y = df_encoded["Disruption_Occurred"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    xgb_model = _nuevo_xgb((len(y_train) - sum(y_train)) / sum(y_train))
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_val)
    y_prob = xgb_model.predict_proba(X_val)[:, 1]
    return xgb_model, X, y_val, y_pred, y_prob


@st.cache_resource(show_spinner="Comparando técnicas de reducción…")
def villanueva_comparar_reduccion(X_all, y_all):
    """Compara las técnicas ajustando SOLO con datos de entrenamiento (sin fuga)."""
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_all.astype(float), y_all, test_size=0.25, random_state=42, stratify=y_all
    )
    n_vars = X_tr.shape[1]

    def reducir(metodo, Xa, ya, Xb):
        if metodo[0] == "none":
            return Xa.values, Xb.values
        sc = StandardScaler().fit(Xa)
        A, B = sc.transform(Xa), sc.transform(Xb)
        if metodo[0] == "pca":
            t = PCA(n_components=metodo[1], svd_solver="full", random_state=42).fit(A)
        elif metodo[0] == "lda":
            t = LinearDiscriminantAnalysis(n_components=1).fit(A, ya)
        else:
            t = SelectKBest(f_classif, k=metodo[1]).fit(A, ya)
        return t.transform(A), t.transform(B)

    def auc_con(metodo, Xa, ya, Xb, yb):
        A, B = reducir(metodo, Xa, ya, Xb)
        m = _nuevo_xgb((len(ya) - ya.sum()) / ya.sum()).fit(A, ya)
        return roc_auc_score(yb, m.predict_proba(B)[:, 1]), A.shape[1]

    skf = StratifiedKFold(5, shuffle=True, random_state=42)

    def auc_cv(metodo):
        aucs, dims = [], []
        for ia, ib in skf.split(X_tr, y_tr):
            a, d = auc_con(metodo, X_tr.iloc[ia], y_tr.iloc[ia], X_tr.iloc[ib], y_tr.iloc[ib])
            aucs.append(a)
            dims.append(d)
        return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(dims))

    ks = sorted(set([k for k in [1, 2, 3, 4, 5, 6, 8, 10, 12] if k < n_vars]))
    curva = [auc_cv(("kbest", k)) for k in ks]
    medias = np.array([c[0] for c in curva])
    mejor = int(np.argmax(medias))
    umbral = medias[mejor] - curva[mejor][1] / np.sqrt(5)
    k_opt = int(next(k for k, m in zip(ks, medias) if m >= umbral))

    metodos = [
        ("Sin reducción", "Referencia", ("none",), "Sí"),
        ("PCA (90 % de varianza)", "No supervisada", ("pca", 0.90), "No (combinaciones)"),
        ("PCA (2 componentes)", "No supervisada", ("pca", 2), "No (combinaciones)"),
        ("LDA (1 componente)", "Supervisada", ("lda",), "No (un solo eje)"),
        (f"SelectKBest ANOVA F (k={k_opt})", "Supervisada (filtro)", ("kbest", k_opt), "Sí"),
    ]
    filas = []
    for nombre, tipo, met, conserva in metodos:
        a_cv, s_cv, dims = auc_cv(met)
        a_ho, _ = auc_con(met, X_tr, y_tr, X_va, y_va)
        filas.append({"Método": nombre, "Tipo": tipo, "Dimensiones": dims, "AUC_CV": a_cv,
                      "STD_CV": s_cv, "AUC_HO": a_ho, "Conserva variables originales": conserva})
    tabla = pd.DataFrame(filas)

    F, pv = f_classif(X_tr, y_tr)
    fscores = (pd.DataFrame({"Variable": X_tr.columns, "F (ANOVA)": F, "p-valor": pv})
               .sort_values("F (ANOVA)", ascending=False).reset_index(drop=True))
    fscores["Seleccionada"] = ["Sí" if i < k_opt else "No" for i in range(len(fscores))]

    cum = np.cumsum(PCA(random_state=42).fit(StandardScaler().fit_transform(X_tr)).explained_variance_ratio_)
    return {"tabla": tabla, "ks": ks, "curva": curva, "k_opt": k_opt, "fscores": fscores,
            "k90": int(np.argmax(cum >= 0.90)) + 1, "n_vars": n_vars}


@st.cache_resource(show_spinner="Evaluación temporal…")
def villanueva_evaluar_temporal(df_input):
    d = df_input.drop(columns=["P_Disruption"], errors="ignore").copy()
    tr = (d["Date"] < V_FECHA_CORTE).values
    enc = pd.get_dummies(d, columns=V_CAT_COLS, drop_first=True)
    base = [c for c in enc.columns if c not in V_COLS_EXCLUIDAS]
    y = enc["Disruption_Occurred"]

    def ajustar(feats):
        ytr = y[tr]
        m = _nuevo_xgb((len(ytr) - ytr.sum()) / ytr.sum())
        m.fit(enc.loc[tr, feats], ytr)
        p = m.predict_proba(enc[feats])[:, 1]
        return p, roc_auc_score(y[~tr], p[~tr])

    d["P_full"], auc_full = ajustar(base)
    d["P_exante"], auc_exante = ajustar([f for f in base if f != "Lead_Time_Days"])
    return d, tr, auc_full, auc_exante


def villanueva_demanda_dataset(df_in, cat, fechas):
    serie = (df_in[df_in["Product_Category"] == cat]
             .groupby("Date")["Weight_MT"].sum()
             .reindex(fechas, fill_value=0))
    return float(serie.mean()), float(serie.std())


def render_villanueva(raw: pd.DataFrame):
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    st.title("🚚 Optimización de la Cadena de Abastecimiento: Clasificación de la Interrupción con XGBoost")
    st.markdown("**Alumno:** Juan Eduardo Villanueva Rios | **Dimensión:** VILLANUEVA (Optimización de la cadena de abastecimiento · Clasificación XGBoost de la disrupción · SS y ROP)")

    xgb_model, X_feats, y_val, y_pred, y_prob = villanueva_train_model(df)
    df["P_Disruption"] = xgb_model.predict_proba(X_feats)[:, 1]

    red = villanueva_comparar_reduccion(X_feats, df["Disruption_Occurred"])
    t_red = red["tabla"]
    base_cv, base_ho = t_red.loc[0, "AUC_CV"], t_red.loc[0, "AUC_HO"]
    pca90_cv, pca2_cv = t_red.loc[1, "AUC_CV"], t_red.loc[2, "AUC_CV"]
    lda_cv, kb_cv, kb_ho = t_red.loc[3, "AUC_CV"], t_red.loc[4, "AUC_CV"], t_red.loc[4, "AUC_HO"]
    pca_es_peor = bool(pca90_cv < min(base_cv, lda_cv, kb_cv) and pca2_cv < min(base_cv, lda_cv, kb_cv))
    vars_sel = red["fscores"].loc[red["fscores"]["Seleccionada"] == "Sí", "Variable"].tolist()

    d_eval, tr_mask, auc_full_t, auc_exante_t = villanueva_evaluar_temporal(df)

    auc_val = roc_auc_score(y_val, y_prob)
    acc_val = accuracy_score(y_val, y_pred)
    prec_val = precision_score(y_val, y_pred)
    rec_val = recall_score(y_val, y_pred)
    f1_val = f1_score(y_val, y_pred)

    # Parámetros en la barra lateral
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Parámetros de Optimización")
    service_level = st.sidebar.slider("Nivel de Servicio Deseado (%)", 80.0, 99.9, 95.0, step=0.5, key="v_sl")
    z_alpha = float(norm.ppf(service_level / 100.0))
    st.sidebar.caption(f"z = {z_alpha:.3f} (cuantil unilateral de la normal estándar)")

    categorias = df["Product_Category"].unique()

    fuente = st.sidebar.radio(
        "Fuente de la demanda",
        ["Dataset (toneladas despachadas por día)", "Ingreso manual (supuesto del usuario)"],
        key="v_fuente",
    )
    usar_dataset = fuente.startswith("Dataset")
    unidad = "MT" if usar_dataset else "unidades"

    demanda_inputs = {}
    st.sidebar.markdown("### 📈 Demanda por Categoría")
    if usar_dataset:
        st.sidebar.caption(
            "El dataset no registra demanda ni inventario. Se usa el peso despachado por día "
            "(Weight_MT agrupado por Date y categoría) como aproximación de la demanda."
        )
        fechas_all = pd.date_range(df["Date"].min(), df["Date"].max())
        for cat in categorias:
            d_mean, d_std = villanueva_demanda_dataset(df, cat, fechas_all)
            demanda_inputs[cat] = {"mean": d_mean, "std": d_std}
            st.sidebar.markdown(f"**{cat}**: {d_mean:,.0f} ± {d_std:,.0f} MT/día")
    else:
        st.sidebar.caption("Valores supuestos por el usuario: no provienen del dataset.")
        for cat in categorias:
            st.sidebar.markdown(f"**{cat}**")
            d_mean = st.sidebar.number_input(f"Demanda Promedio ({cat})", 50, 500, value=200, key=f"v_{cat}_m")
            d_std = st.sidebar.number_input(f"Desv. Estándar ({cat})", 5, 100, value=30, key=f"v_{cat}_s")
            demanda_inputs[cat] = {"mean": d_mean, "std": d_std}

    st.subheader("🎯 Clasificación de la interrupción con XGBoost")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AUC-ROC", f"{auc_val:.2f}", help=f"Valor exacto: {auc_val:.4f}")
    m2.metric("Exactitud", f"{acc_val:.2%}")
    m3.metric("Precisión", f"{prec_val:.2%}")
    m4.metric("Sensibilidad (Recall)", f"{rec_val:.2%}")
    m5.metric("F1-Score", f"{f1_val:.2%}")
    st.caption(
        f"El XGBoost clasifica si un envío sufrirá una disrupción con un AUC de {auc_val:.4f} "
        "sobre el 25% de validación. Esa probabilidad se usa para ajustar el Lead Time y optimizar SS y ROP."
    )

    tab1, tab2, tab3, tab5, tab4 = st.tabs([
        "📊 Optimización de Inventario",
        "📉 Reducción Dimensional (ANOVA F vs PCA vs LDA)",
        "📈 Rendimiento del Modelo",
        "✅ Evaluación de la Optimización",
        "📝 Marco Teórico y Conclusiones",
    ])

    # TAB 1
    with tab1:
        st.header("Calibración de Stock de Seguridad (SS) y Punto de Reorden (ROP)")
        st.markdown('''
        Este módulo calcula las métricas de reabastecimiento adaptando el **Lead Time nominal** según la 
        probabilidad matemática de riesgo de disrupción provista por el modelo predictivo de Machine Learning:
        ''')

        inventory_results = []
        for cat in categorias:
            df_cat = df[df["Product_Category"] == cat]
            L_mean = df_cat["Lead_Time_Days"].mean()
            L_std = df_cat["Lead_Time_Days"].std() if df_cat["Lead_Time_Days"].std() > 0 else 0.5
            avg_p_risk = df_cat["P_Disruption"].mean()

            L_hat = L_mean * (1 + avg_p_risk)
            D_mean = demanda_inputs[cat]["mean"]
            D_std = demanda_inputs[cat]["std"]

            term1 = (D_mean ** 2) * (L_std ** 2)
            term2 = (L_hat ** 2) * (D_std ** 2)
            SS = z_alpha * np.sqrt(term1 + term2)
            ROP = (D_mean * L_hat) + SS

            inventory_results.append({
                "Categoría": cat,
                "Prob. Riesgo Promedio": f"{avg_p_risk:.2%}",
                "Lead Time Original (días)": round(L_mean, 2),
                "Lead Time Ajustado (días)": round(L_hat, 2),
                f"Demanda diaria prom. ({unidad})": int(D_mean),
                f"Stock de Seguridad (SS) [{unidad}]": int(SS),
                f"Punto de Reorden (ROP) [{unidad}]": int(ROP),
            })

        df_inventory = pd.DataFrame(inventory_results)
        st.dataframe(df_inventory, use_container_width=True)

        st.subheader("Comparación Gráfica de Métricas de Inventario")
        df_melted = df_inventory.melt(
            id_vars=["Categoría"],
            value_vars=[f"Stock de Seguridad (SS) [{unidad}]", f"Punto de Reorden (ROP) [{unidad}]"],
            var_name="Métrica", value_name="Cantidad",
        )
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.barplot(data=df_melted, x="Categoría", y="Cantidad", hue="Métrica",
                    palette=["#aec7e8", "#1f77b4"], edgecolor="black", ax=ax)
        ax.set_ylabel(f"Cantidad ({unidad})")
        ax.set_title("Comparativa de SS y ROP Mitigado por Categoría de Producto")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        st.pyplot(fig)

    # TAB 2
    with tab2:
        st.header("📉 Reducción de Dimensionalidad: ¿PCA, LDA o SelectKBest (ANOVA F)?")
        st.markdown(f'''
        El objetivo es **clasificar una variable binaria** (`Disruption_Occurred`) con XGBoost a partir de {red['n_vars']} variables 
        (numéricas y categóricas codificadas). Se compararon tres técnicas con el mismo modelo, la misma partición del informe (75/25) 
        y validación cruzada de 5 particiones; cada técnica se ajusta **solo con los datos de entrenamiento**.
        ''')

        tabla_vis = t_red.copy()
        tabla_vis["Dimensiones"] = tabla_vis["Dimensiones"].round(1)
        tabla_vis["AUC validación cruzada"] = [f"{a:.4f} ± {s_:.4f}" for a, s_ in zip(t_red["AUC_CV"], t_red["STD_CV"])]
        tabla_vis["AUC hold-out (25 %)"] = t_red["AUC_HO"].map(lambda v: f"{v:.4f}")
        st.dataframe(
            tabla_vis[["Método", "Tipo", "Dimensiones", "AUC validación cruzada",
                       "AUC hold-out (25 %)", "Conserva variables originales"]],
            use_container_width=True, hide_index=True,
        )

        col_r1, col_r2 = st.columns(2)
        fs = red["fscores"]
        with col_r1:
            st.subheader("Puntaje F (ANOVA) por variable")
            fig_f, ax_f = plt.subplots(figsize=(6, 5.5))
            colores = ["#1f77b4" if v == "Sí" else "#c7c7c7" for v in fs["Seleccionada"]]
            ax_f.barh(fs["Variable"][::-1], fs["F (ANOVA)"][::-1], color=colores[::-1], edgecolor="black")
            ax_f.set_xlabel("F (ANOVA)")
            ax_f.set_title(f"Variables seleccionadas (azul): {red['k_opt']} de {red['n_vars']}")
            ax_f.grid(axis="x", linestyle="--", alpha=0.5)
            st.pyplot(fig_f)

        with col_r2:
            st.subheader("AUC según el número de variables (k)")
            medias_k = np.array([c[0] for c in red["curva"]])
            stds_k = np.array([c[1] for c in red["curva"]])
            fig_k, ax_k = plt.subplots(figsize=(6, 5.5))
            ax_k.errorbar(red["ks"], medias_k, yerr=stds_k, marker="o", color="#1f77b4",
                          capsize=3, label="SelectKBest (ANOVA F)")
            ax_k.axhline(base_cv, color="gray", linestyle="--", label=f"Sin reducción ({red['n_vars']} variables)")
            ax_k.axvline(red["k_opt"], color="#d62728", linestyle=":", label=f"k elegido = {red['k_opt']}")
            ax_k.set_xlabel("Número de variables conservadas (k)")
            ax_k.set_ylabel("AUC (validación cruzada)")
            ax_k.legend(loc="lower right")
            ax_k.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig_k)

        st.subheader("Puntajes F y p-valores (conjunto de entrenamiento)")
        fs_vis = fs.copy()
        fs_vis["F (ANOVA)"] = fs_vis["F (ANOVA)"].round(1)
        fs_vis["p-valor"] = fs_vis["p-valor"].map(lambda v: f"{v:.2e}")
        st.dataframe(fs_vis, use_container_width=True, hide_index=True)

        st.markdown(f'''
        **Conclusión (valores calculados sobre los datos):**
        * **{'PCA no es la mejor opción para este problema.' if pca_es_peor else 'PCA no muestra ventaja frente a las otras técnicas en este problema.'}** Es no supervisada: ordena por varianza, no por relación con la disrupción. Necesita {red['k90']} de {red['n_vars']} componentes para llegar al 90 % de varianza (casi no reduce) y, aun así, su AUC es {pca90_cv:.4f} frente a {base_cv:.4f} sin reducir; con 2 componentes baja a {pca2_cv:.4f}.
        * **LDA** con dos clases solo genera **1 componente** (AUC {lda_cv:.4f}). Comprime al máximo, pero reemplaza todas las variables por un único eje lineal: se pierde la interpretabilidad y la estructura no lineal que justifica usar XGBoost.
        * **SelectKBest con ANOVA F (técnica adoptada):** conserva {red['k_opt']} de {red['n_vars']} variables originales ({', '.join(vars_sel)}), con AUC de {kb_cv:.4f} en validación cruzada y {kb_ho:.4f} en el hold-out (todas las variables: {base_ho:.4f}). Mantiene el significado operativo de cada variable y el p-valor justifica descartar el resto.
        * **Cautela:** las diferencias entre «sin reducción», LDA y ANOVA F están dentro de la variabilidad de la validación cruzada (± {t_red.loc[0, 'STD_CV']:.3f}); {'lo claro es que PCA es el único que pierde capacidad predictiva sin reducir de forma útil.' if pca_es_peor else 'revisa la tabla para ver cuál técnica rinde mejor.'} XGBoost no necesita reducción para predecir bien: el beneficio es parsimonia e interpretabilidad.
        * **Limitación de ANOVA F:** evalúa cada variable por separado y de forma lineal, por lo que no detecta interacciones ni redundancia entre variables (por ejemplo, entre clima de huracán y Lead Time).
        ''')

    # TAB 3
    with tab3:
        st.header("📈 Evaluación de Capacidad Predictiva del Modelo XGBoost")
        st.markdown('''
        XGBoost clasifica de manera probabilística si una orden sufrirá demoras extremas o bloqueos 
        de tránsito basándose en el historial multivariante del envío.
        ''')

        col_eval1, col_eval2 = st.columns(2)
        with col_eval1:
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            ax_roc.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"Curva ROC (AUC = {roc_auc:.4f})")
            ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel("Tasa de Falsos Positivos")
            ax_roc.set_ylabel("Tasa de Verdaderos Positivos")
            ax_roc.set_title("Curva ROC")
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig_roc)

        with col_eval2:
            cm = confusion_matrix(y_val, y_pred, normalize="true")
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disrupción", "Disrupción"])
            disp.plot(ax=ax_cm, cmap="Blues", values_format=".2%")
            ax_cm.set_title("Matriz de Confusión Normalizada")
            ax_cm.grid(False)
            st.pyplot(fig_cm)

        st.subheader("Métricas de clasificación (conjunto de validación)")
        df_metrics = pd.DataFrame({
            "Métrica": ["AUC-ROC", "Exactitud", "Precisión", "Sensibilidad (Recall)", "F1-Score"],
            "Valor": [f"{auc_val:.4f}", f"{acc_val:.2%}", f"{prec_val:.2%}", f"{rec_val:.2%}", f"{f1_val:.2%}"],
        })
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # TAB 5
    with tab5:
        st.header("✅ ¿Se logró optimizar la cadena de abastecimiento con este dataset?")
        st.markdown(
            "Esta pestaña mide el resultado usando **solo columnas del dataset**. "
            f"Entrenamiento y parámetros: envíos anteriores al {V_FECHA_CORTE.date()}; "
            f"prueba: envíos desde esa fecha ({int((~tr_mask).sum())} de {len(d_eval)} envíos)."
        )

        st.subheader("1. Qué permite (y qué no) evaluar el dataset")
        st.dataframe(pd.DataFrame({
            "Elemento": ["Lead Time por envío", "Ocurrencia de disrupción", "Fecha del envío",
                         "Peso despachado (Weight_MT)",
                         "Demanda real, costos de inventario/faltante/flete, nivel de inventario, OTIF"],
            "¿Está en el dataset?": ["Sí (Lead_Time_Days)", "Sí (Disruption_Occurred)",
                                     f"Sí ({df['Date'].min().date()} a {df['Date'].max().date()})",
                                     "Sí (se usa como aproximación de demanda)", "No"],
            "Consecuencia": ["Base del cálculo de cobertura", "Variable objetivo del XGBoost",
                             "Permite validar en el tiempo (entrenar con el pasado, probar con el futuro)",
                             "Reemplaza la demanda supuesta de 200 ± 30",
                             "No se puede afirmar ahorro económico ni mejora de OTIF; solo cobertura de Lead Time y nivel de servicio por ciclo"],
        }), use_container_width=True, hide_index=True)

        st.subheader("2. Calidad de la probabilidad que alimenta la optimización")
        tasa_obs = float(y_val.mean())
        p_medio = float(np.mean(y_prob))
        spw_usado = (1 - df["Disruption_Occurred"].mean()) / df["Disruption_Occurred"].mean()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prob. media predicha (validación)", f"{p_medio:.1%}")
        c2.metric("Tasa real de disrupción (validación)", f"{tasa_obs:.1%}")
        c3.metric("AUC temporal con Lead_Time_Days", f"{auc_full_t:.3f}")
        c4.metric("AUC temporal sin Lead_Time_Days", f"{auc_exante_t:.3f}")
        st.caption(
            f"scale_pos_weight usado = {spw_usado:.2f}. Si la disrupción es la clase **mayoritaria** "
            f"({df['Disruption_Occurred'].mean():.1%} de los envíos), ese peso es menor que 1 y las probabilidades "
            "salen por debajo de la frecuencia real: el Lead Time ajustado queda subestimado. "
            "Además, Lead_Time_Days solo se conoce cuando el envío llegó; el modelo sin esa variable es el que se podría usar antes de despachar."
        )

        st.subheader(f"3. Prueba retrospectiva por categoría (objetivo de servicio {service_level:.1f}%)")
        d_tr = d_eval[tr_mask]
        d_te = d_eval[~tr_mask]
        fechas_tr = pd.date_range(d_tr["Date"].min(), d_tr["Date"].max())
        q = service_level / 100.0

        filas = []
        for cat in categorias:
            dtr = d_tr[d_tr["Product_Category"] == cat]
            dte = d_te[d_te["Product_Category"] == cat]
            D, sD = villanueva_demanda_dataset(d_tr, cat, fechas_tr)
            L, sL = dtr["Lead_Time_Days"].mean(), dtr["Lead_Time_Days"].std()
            L_hat = L * (1 + dtr["P_full"].mean())

            def cobertura(Lx, D=D, sD=sD, sL=sL):
                ss = z_alpha * np.sqrt(D ** 2 * sL ** 2 + Lx ** 2 * sD ** 2)
                return Lx + ss / D

            politicas = {
                "A. Sin ajuste por riesgo": cobertura(L),
                "B. Con ajuste por riesgo (modelo del informe)": cobertura(L_hat),
                "C. Percentil empírico del Lead Time": float(dtr["Lead_Time_Days"].quantile(q)),
            }
            for nombre, cob in politicas.items():
                ok = int((dte["Lead_Time_Days"] <= cob).sum())
                filas.append({"Categoría": cat, "Política": nombre, "Cobertura (días)": cob,
                              "ROP (MT)": D * cob, "ok": ok, "n": len(dte)})
        df_bt = pd.DataFrame(filas)
        df_bt["Servicio logrado"] = df_bt["ok"] / df_bt["n"]

        st.dataframe(
            df_bt.assign(**{
                "Cobertura (días)": df_bt["Cobertura (días)"].round(1),
                "ROP (MT)": df_bt["ROP (MT)"].round(0).astype(int),
                "Servicio logrado": (df_bt["Servicio logrado"] * 100).round(1).astype(str) + "%",
            })[["Categoría", "Política", "Cobertura (días)", "ROP (MT)", "Servicio logrado"]],
            use_container_width=True, hide_index=True,
        )

        resumen = df_bt.groupby("Política").agg(ok=("ok", "sum"), n=("n", "sum"), rop=("ROP (MT)", "sum"))
        resumen["servicio"] = resumen["ok"] / resumen["n"]
        sA = resumen.loc["A. Sin ajuste por riesgo", "servicio"]
        sB = resumen.loc["B. Con ajuste por riesgo (modelo del informe)", "servicio"]
        sC = resumen.loc["C. Percentil empírico del Lead Time", "servicio"]
        rA = resumen.loc["A. Sin ajuste por riesgo", "rop"]
        rB = resumen.loc["B. Con ajuste por riesgo (modelo del informe)", "rop"]
        rC = resumen.loc["C. Percentil empírico del Lead Time", "rop"]

        st.dataframe(pd.DataFrame({
            "Política": resumen.index,
            "Servicio logrado (total)": [f"{v:.1%}" for v in resumen["servicio"]],
            "ROP total (MT)": [f"{v:,.0f}" for v in resumen["rop"]],
            "ROP vs. política A": [f"{v / rA - 1:+.1%}" for v in resumen["rop"]],
        }), use_container_width=True, hide_index=True)
        st.caption(
            "Servicio logrado = proporción de envíos de prueba cuyo Lead Time real cabe dentro de la cobertura "
            "(ROP / demanda diaria). La demanda es la aproximación por Weight_MT; no se asumen costos."
        )

        st.subheader("4. Cobertura segmentada por riesgo predicho (modelo sin Lead_Time_Days)")
        cortes = np.quantile(d_tr["P_exante"], [0.25, 0.5, 0.75])
        d_eval = d_eval.assign(Segmento=np.digitize(d_eval["P_exante"], cortes))
        d_tr = d_eval[tr_mask]
        d_te = d_eval[~tr_mask]
        nombres_seg = {0: "Riesgo bajo", 1: "Riesgo medio-bajo", 2: "Riesgo medio-alto", 3: "Riesgo alto"}

        cob_global = float(d_tr["Lead_Time_Days"].quantile(q))
        cob_seg = d_tr.groupby("Segmento")["Lead_Time_Days"].quantile(q)
        d_te = d_te.assign(cob_seg=d_te["Segmento"].map(cob_seg))
        serv_global = float((d_te["Lead_Time_Days"] <= cob_global).mean())
        serv_seg = float((d_te["Lead_Time_Days"] <= d_te["cob_seg"]).mean())
        cob_prom_seg = float(d_te["cob_seg"].mean())

        tabla_seg = []
        for s in range(4):
            sub = d_te[d_te["Segmento"] == s]
            tabla_seg.append({
                "Segmento": nombres_seg[s],
                "Envíos (prueba)": len(sub),
                "Disrupción observada": f"{sub['Disruption_Occurred'].mean():.1%}",
                "Lead Time medio (días)": round(sub["Lead_Time_Days"].mean(), 1),
                "Cobertura segmentada (días)": round(float(cob_seg[s]), 1),
                "Cobertura global (días)": round(cob_global, 1),
                "Servicio logrado": f"{(sub['Lead_Time_Days'] <= sub['cob_seg']).mean():.1%}",
            })
        st.dataframe(pd.DataFrame(tabla_seg), use_container_width=True, hide_index=True)

        fig_seg, ax_seg = plt.subplots(figsize=(8, 3.8))
        ax_seg.bar([nombres_seg[s] for s in range(4)], [cob_seg[s] for s in range(4)],
                   color="#1f77b4", edgecolor="black")
        ax_seg.axhline(cob_global, color="#d62728", linestyle="--",
                       label=f"Cobertura global única ({cob_global:.1f} días)")
        ax_seg.set_ylabel("Cobertura (días)")
        ax_seg.set_title("Cobertura necesaria por segmento de riesgo vs. política global")
        ax_seg.legend()
        ax_seg.grid(axis="y", linestyle="--", alpha=0.6)
        st.pyplot(fig_seg)

        st.subheader("5. Veredicto con los datos")
        if sA >= q - 0.01:
            juicio_B = (f"La política A ya alcanza el objetivo sin usar el modelo, por lo que el ajuste por riesgo de la política B "
                        f"**no era necesario para cumplir el servicio**; solo aumentó el inventario requerido ({rB / rA - 1:+.1%}).")
        else:
            juicio_B = (f"La política A queda por debajo del objetivo y el ajuste por riesgo (B) lo acerca "
                        f"con {rB / rA - 1:+.1%} de ROP.")
        pmin = d_tr.groupby("Product_Category")["P_full"].mean().min()
        pmax = d_tr.groupby("Product_Category")["P_full"].mean().max()
        st.markdown(f'''
        * **Lo que sí se logró:** el XGBoost clasifica la disrupción con AUC {auc_val:.4f} y se integra a la fórmula de SS y ROP.
        * **Ajuste por categoría:** con objetivo de {service_level:.1f}%, la política A logró {sA:.1%} en la prueba y la B {sB:.1%}; la referencia empírica C logró {sC:.1%} con {rC / rA - 1:+.1%} de ROP respecto a A. {juicio_B}
        * **Por qué el ajuste por categoría aporta poco:** la probabilidad media de disrupción casi no cambia entre categorías ({pmin:.1%} a {pmax:.1%}), así que promediarla por categoría descarta el poder discriminante del modelo (que opera a nivel de envío).
        * **Dónde sí hay optimización:** al segmentar por riesgo predicho antes del despacho, la cobertura promedio baja de {cob_global:.1f} a {cob_prom_seg:.1f} días ({cob_prom_seg / cob_global - 1:+.1%}) con servicio de {serv_seg:.1%} frente a {serv_global:.1%} de la política global.
        * **Límite del dataset:** no contiene costos, inventario real, demanda ni OTIF. Por eso no se puede cuantificar ahorro monetario; el logro se mide en días de cobertura y nivel de servicio por ciclo.
        ''')

    # TAB 4
    with tab4:
        st.header("📘 Fundamentación Teórica e Impacto Organizacional")

        st.subheader("1. Formulación del Modelo Estocástico")
        st.markdown(r'''
        En operaciones con alta variabilidad, el Stock de Seguridad ($SS$) se calcula mediante la fórmula estocástica de interacción:

        $$SS = z_{\alpha} \cdot \sqrt{\bar{D}^2 \cdot \sigma_L^2 + \hat{L}^2 \cdot \sigma_D^2}$$

        Donde el Lead Time corregido por riesgo logístico de disrupción es:

        $$\hat{L} = \bar{L} \cdot (1 + P(\text{Disruption}))$$

        Esta corrección penaliza el tiempo de ciclo promedio, incrementando los márgenes de seguridad en categorías con mayor probabilidad de disrupción. Aquí $z_{\alpha}$ es el cuantil unilateral de la normal para el nivel de servicio elegido.
        ''')

        st.subheader("2. Decisiones y Conclusiones Operativas")
        st.markdown(f'''
        * **Clasificación de la disrupción:** el XGBoost alcanza un AUC de {auc_val:.4f}: asigna mayor probabilidad a un envío que sí sufrirá disrupción que a uno normal en ese porcentaje de los pares comparados.
        * **Uso del modelo en el inventario:** la probabilidad predicha ajusta el Lead Time y, con él, el SS y el ROP. La pestaña de evaluación muestra que el mayor beneficio medible aparece al **segmentar la cobertura por riesgo predicho**, no al promediar el riesgo por categoría.
        * **Reducción de dimensionalidad:** se comparó PCA, LDA y SelectKBest (ANOVA F) con el mismo clasificador; {'PCA fue la técnica menos adecuada y ' if pca_es_peor else ''}se adopta ANOVA F por conservar variables originales interpretables, que conserva {red['k_opt']} de {red['n_vars']} variables originales.
        * **Alcance de las conclusiones:** el dataset no incluye costos, inventario, demanda ni OTIF; por eso no se afirman ahorros ni mejoras financieras, y la demanda se aproxima con el peso despachado por día.
        ''')


# =============================================================================
# 4. DIMENSIÓN HUAPAYA — Integración con Power BI y R
# =============================================================================
H_NUMERICAS = [
    "Distance_km", "Weight_MT", "Fuel_Price_Index",
    "Geopolitical_Risk_Score", "Carrier_Reliability_Score", "Lead_Time_Days",
]
H_CATEGORICAS = ["Transport_Mode", "Weather_Condition"]


def huapaya_preparar_matriz(df: pd.DataFrame):
    X = pd.get_dummies(df[H_NUMERICAS + H_CATEGORICAS], columns=H_CATEGORICAS, drop_first=True)
    return X, df["Disruption_Occurred"]


def huapaya_datos_entrada(df: pd.DataFrame):
    st.subheader("1. Consideraciones de datos de entrada al modelo")
    st.write(
        "Análisis descriptivo de las variables logísticas, económicas y "
        "geopolíticas del envío, y de la variable ambiental (clima), que "
        "alimentan el modelo de predicción de disrupciones."
    )

    desc = df[H_NUMERICAS].describe().T[["mean", "std", "min", "max"]]
    desc["mode"] = df[H_NUMERICAS].mode().iloc[0]
    desc = desc[["mean", "mode", "std", "min", "max"]].round(2)
    desc.columns = ["Media", "Moda", "Desv. estándar", "Mínimo", "Máximo"]
    st.dataframe(desc, use_container_width=True)

    st.metric("Tasa global de disrupción en el dataset", f"{df['Disruption_Occurred'].mean():.1%}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        tasa_clima = (df.groupby("Weather_Condition")["Disruption_Occurred"]
                      .mean().sort_values(ascending=False))
        ax.bar(tasa_clima.index, tasa_clima.values, color="#C0392B")
        ax.set_ylabel("Tasa de disrupción")
        ax.set_title("Disrupción por condición ambiental (clima)")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        corr = (df[H_NUMERICAS + ["Disruption_Occurred"]].corr()["Disruption_Occurred"]
                .drop("Disruption_Occurred").sort_values())
        colores = ["#2E5A88" if v >= 0 else "#B5651D" for v in corr.values]
        ax.barh(corr.index, corr.values, color=colores)
        ax.set_title("Correlación (Pearson) con la disrupción")
        ax.axvline(0, color="black", linewidth=0.8)
        st.pyplot(fig)

    st.caption(
        "El clima (variable ambiental) muestra la mayor asociación categórica "
        "con la disrupción, mientras que el riesgo geopolítico y el lead time "
        "presentan las correlaciones numéricas más altas; la distancia, el peso "
        "y el índice de combustible muestran una relación numérica débil por sí solos."
    )
    huapaya_pca(df)


def huapaya_pca(df: pd.DataFrame):
    st.markdown("**Reducción de dimensiones — Análisis de Componentes Principales (PCA)**")

    X, y = huapaya_preparar_matriz(df)
    X_esc = StandardScaler().fit_transform(X)
    pca = PCA(n_components=X.shape[1], random_state=42)
    componentes = pca.fit_transform(X_esc)

    varianza = pca.explained_variance_ratio_
    acumulada = np.cumsum(varianza)
    n_componentes_90 = int(np.argmax(acumulada >= 0.90) + 1)

    k1, k2, k3 = st.columns(3)
    k1.metric("Variables originales", f"{X.shape[1]}")
    k2.metric("Componentes para 90% de varianza", f"{n_componentes_90}")
    k3.metric("Varianza explicada (2 primeros PC)", f"{acumulada[1]:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.bar(range(1, len(varianza) + 1), varianza, color="#6A2C70", label="Individual")
        ax.plot(range(1, len(varianza) + 1), acumulada, "o-", color="#C0392B", label="Acumulada")
        ax.axhline(0.90, color="gray", linestyle="--", linewidth=1, label="Umbral 90%")
        ax.set_xlabel("Componente principal")
        ax.set_ylabel("Varianza explicada")
        ax.set_title("Varianza explicada por componente (scree plot)")
        ax.legend(fontsize=7)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        scatter = ax.scatter(componentes[:, 0], componentes[:, 1], c=y, cmap="coolwarm", alpha=0.5, s=12)
        ax.set_xlabel(f"PC1 ({varianza[0]:.1%} varianza)")
        ax.set_ylabel(f"PC2 ({varianza[1]:.1%} varianza)")
        ax.set_title("Envíos proyectados en PC1 y PC2")
        leg = ax.legend(*scatter.legend_elements(), title="Disrupción", fontsize=7)
        ax.add_artist(leg)
        st.pyplot(fig)

    st.caption(
        f"Con solo **{n_componentes_90} componentes de las {X.shape[1]} variables originales** "
        f"se explica el 90% de la varianza del conjunto de datos de entrada. Esto indica el grado de "
        f"redundancia/correlación entre variables y si el problema puede tratarse en un espacio de "
        f"menor dimensión antes de -o en paralelo a- la regularización L1 del modelo."
    )


def huapaya_modelo(df: pd.DataFrame):
    st.subheader("2. Características del modelo de predicción")
    st.write(
        "Se comparan dos algoritmos supervisados de clasificación binaria para "
        "estimar `Disruption_Occurred`, consistentes con el fundamento "
        "estadístico de una variable respuesta binomial:"
    )
    st.markdown(
        "- **Regresión Logística con regularización L1 (Lasso logístico):** "
        "modelo lineal generalizado que estima la probabilidad de disrupción, "
        "con selección automática de variables mediante penalización L1.\n"
        "- **Random Forest:** ensamble no paramétrico de árboles de decisión, "
        "usado como contraste no lineal frente al modelo logístico."
    )

    X, y = huapaya_preparar_matriz(df)

    col1, col2, col3 = st.columns(3)
    test_size = col1.slider("Proporción de prueba (test)", 0.15, 0.4, 0.25, key="huapaya_test_size")
    c_logit = col2.slider("Fuerza de regularización L1 (C, a menor valor más variables se eliminan)",
                          0.05, 2.0, 0.5, key="huapaya_c_logit")
    n_arboles = col3.slider("N° de árboles del Random Forest", 50, 400, 200, step=50, key="huapaya_n_arboles")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    escalador = StandardScaler()
    Xtr_s = escalador.fit_transform(Xtr)
    Xte_s = escalador.transform(Xte)

    logit = LogisticRegression(penalty="l1", solver="liblinear", C=c_logit, random_state=42).fit(Xtr_s, ytr)
    rf = RandomForestClassifier(n_estimators=n_arboles, max_depth=6, random_state=42).fit(Xtr, ytr)

    modelos = {
        "Regresión Logística (L1)": (logit, Xte_s),
        "Random Forest": (rf, Xte),
    }

    filas, curvas_roc = [], {}
    for nombre, (modelo, Xte_m) in modelos.items():
        pred = modelo.predict(Xte_m)
        proba = modelo.predict_proba(Xte_m)[:, 1]
        filas.append({
            "Modelo": nombre,
            "Exactitud": accuracy_score(yte, pred),
            "Precisión": precision_score(yte, pred),
            "Sensibilidad (Recall)": recall_score(yte, pred),
            "F1-score": f1_score(yte, pred),
            "AUC-ROC": roc_auc_score(yte, proba),
            "R²": r2_score(yte, proba),
        })
        curvas_roc[nombre] = roc_curve(yte, proba)

    tabla_metricas = pd.DataFrame(filas).set_index("Modelo").round(3)
    st.dataframe(tabla_metricas, use_container_width=True)

    st.markdown("**Coeficiente de determinación (R²) de cada modelo**")
    cols_r2 = st.columns(len(tabla_metricas))
    for col_r2, (nombre, fila) in zip(cols_r2, tabla_metricas.iterrows()):
        col_r2.metric(f"R² — {nombre}", f"{fila['R²']:.3f}")
    st.caption(
        "R² = 1 − SC_residual / SC_total, calculado con la probabilidad de disrupción "
        "predicha frente a la variable real (0/1) en el conjunto de prueba. En "
        "clasificación binaria equivale a 1 − Brier/Var(y); valores bajos o negativos "
        "son normales y no invalidan el AUC-ROC."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Curvas ROC**")
        fig, ax = plt.subplots(figsize=(5, 4.2))
        for nombre, (fpr, tpr, _) in curvas_roc.items():
            ax.plot(fpr, tpr, label=nombre)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.legend(fontsize=8)
        st.pyplot(fig)

    with col_b:
        st.markdown("**Variables seleccionadas por regularización L1** (coeficiente ≠ 0)")
        coef = pd.Series(logit.coef_[0], index=X.columns)
        coef_sel = coef[coef != 0].sort_values(key=abs, ascending=False)
        fig, ax = plt.subplots(figsize=(5, 4.2))
        colores = ["#2E5A88" if v >= 0 else "#B5651D" for v in coef_sel.values]
        ax.barh(coef_sel.index, coef_sel.values, color=colores)
        ax.invert_yaxis()
        ax.set_title(f"{len(coef_sel)} de {len(coef)} variables retenidas")
        st.pyplot(fig)

    mejor = tabla_metricas["AUC-ROC"].idxmax()
    st.info(
        f"**Respuesta a la pregunta de investigación:** con las características "
        f"logísticas, ambientales, económicas y geopolíticas disponibles, el "
        f"mejor modelo (**{mejor}**) alcanza un AUC-ROC de "
        f"**{tabla_metricas.loc[mejor, 'AUC-ROC']:.3f}** y una exactitud de "
        f"**{tabla_metricas.loc[mejor, 'Exactitud']:.1%}**. Revisa la tabla y el "
        f"gráfico de coeficientes para ver qué variables concentran el poder predictivo."
    )
    return modelos, X.columns, escalador


def huapaya_script_r(c_logit: float, n_arboles: int) -> str:
    return f"""# ============================================================
# Script R equivalente - Dimensión HUAPAYA
# Predicción de disrupciones en la cadena de suministro
# Pensado para ejecutarse como R script visual / Power Query en Power BI,
# o de forma independiente en R Studio.
# ============================================================
library(glmnet)       # Regresión logística con regularización L1
library(randomForest)  # Random Forest
library(pROC)          # Curvas ROC / AUC

datos <- read.csv("global_supply_chain_risk_2026.csv", stringsAsFactors = TRUE)
datos$Disruption_Occurred <- as.factor(datos$Disruption_Occurred)

variables <- c("Distance_km", "Weight_MT", "Fuel_Price_Index",
               "Geopolitical_Risk_Score", "Carrier_Reliability_Score",
               "Lead_Time_Days", "Transport_Mode", "Weather_Condition")

set.seed(42)
idx <- sample(seq_len(nrow(datos)), size = 0.75 * nrow(datos))
train <- datos[idx, ]
test  <- datos[-idx, ]

X_train <- model.matrix(~ . - 1, data = train[, variables])
X_test  <- model.matrix(~ . - 1, data = test[, variables])
y_train <- train$Disruption_Occurred
y_test  <- test$Disruption_Occurred

# --- Regresión logística con regularización L1 (equivalente a C = {c_logit}) ---
# Usamos glmnet() (no cv.glmnet) porque fijamos un único lambda.
lambda_fijo <- 1 / {c_logit}
modelo_logit <- glmnet(X_train, y_train, family = "binomial",
                        alpha = 1, lambda = lambda_fijo)
pred_logit <- predict(modelo_logit, X_test, type = "response", s = lambda_fijo)

# --- Random Forest ---
modelo_rf <- randomForest(y_train ~ ., data = as.data.frame(X_train),
                           ntree = {n_arboles}, maxnodes = 2^6)
pred_rf <- predict(modelo_rf, as.data.frame(X_test), type = "prob")[, 2]

# --- Evaluación ---
roc_logit <- roc(as.numeric(as.character(y_test)), as.numeric(pred_logit))
roc_rf    <- roc(as.numeric(as.character(y_test)), pred_rf)
cat("AUC Logit L1:", auc(roc_logit), "\\n")
cat("AUC Random Forest:", auc(roc_rf), "\\n")

# --- Salida lista para Power BI ---
salida <- data.frame(test[, c("Shipment_ID", "Origin_Port", "Destination_Port")],
                      Prob_Disrupcion_Logit = as.numeric(pred_logit),
                      Prob_Disrupcion_RF = pred_rf)
write.csv(salida, "huapaya_predicciones_powerbi.csv", row.names = FALSE)
"""


def huapaya_integracion(df, modelos, escalador, c_logit, n_arboles):
    st.subheader("3. Integración de herramientas: R y Power BI")
    st.write(
        "Esta dimensión define **cómo se conecta** el resultado del modelo predictivo con las "
        "herramientas de analítica del equipo: R como motor estadístico y "
        "Power BI como capa de visualización/monitoreo."
    )

    st.markdown("**a) Arquitectura de integración**")
    st.markdown(
        "- **Origen de datos:** `global_supply_chain_risk_2026.csv` "
        "(envíos con sus características logísticas, ambientales, económicas y geopolíticas).\n"
        "- **Motor de modelado (R):** `glmnet` (logística L1) y `randomForest`, "
        "ejecutados en R Studio o como *R script* dentro de Power BI Desktop.\n"
        "- **Capa de visualización (Power BI):** el dataset de predicciones se "
        "consume desde *Obtener datos* o se recalcula en vivo mediante un "
        "**R script visual**, mostrando el riesgo de disrupción por ruta, "
        "modo de transporte y condición climática.\n"
        "- **Requisito técnico:** habilitar R en *Archivo > Opciones y configuración > "
        "Opciones > Scripting de R*, apuntando al motor R instalado localmente "
        "(Power BI Desktop no incluye R por defecto)."
    )

    st.markdown("**b) Bosquejo de script R equivalente al modelo de Python**")
    codigo_r = huapaya_script_r(c_logit, n_arboles)
    st.code(codigo_r, language="r")
    st.download_button("⬇️ Descargar script R (huapaya_modelo.R)", codigo_r.encode("utf-8"),
                       file_name="huapaya_modelo.R", key="huapaya_download_r")

    st.markdown("**c) Exportar predicciones para el tablero de Power BI**")
    logit_modelo, _ = modelos["Regresión Logística (L1)"]
    rf_modelo, _ = modelos["Random Forest"]
    X_full, _ = huapaya_preparar_matriz(df)
    X_full_s = escalador.transform(X_full)

    salida = df[["Shipment_ID", "Origin_Port", "Destination_Port", "Transport_Mode",
                 "Weather_Condition", "Disruption_Occurred"]].copy()
    salida["Prob_Disrupcion_Logit"] = logit_modelo.predict_proba(X_full_s)[:, 1].round(3)
    salida["Prob_Disrupcion_RF"] = rf_modelo.predict_proba(X_full)[:, 1].round(3)

    st.dataframe(salida.head(10), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Exportar CSV para Power BI (huapaya_predicciones_powerbi.csv)",
                       salida.to_csv(index=False).encode("utf-8"),
                       file_name="huapaya_predicciones_powerbi.csv", key="huapaya_download_csv")


def render_huapaya(raw: pd.DataFrame):
    st.title("🧩 Dimensión HUAPAYA — Integración de herramientas con Power BI y R")

    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Disruption_Occurred"] = df["Disruption_Occurred"].astype(int)

    with st.expander("Problema, pregunta y objetivo específico de esta dimensión", expanded=False):
        st.write(
            "**Problema:** la cadena de suministro internacional presenta "
            "incertidumbre y variabilidad en su desempeño, asociadas a cambios "
            "en la demanda de servicios, condiciones de transporte, "
            "disponibilidad/abastecimiento y factores externos, lo que dificulta "
            "la planificación y gestión eficiente de las operaciones."
        )
        st.write(
            "**Pregunta:** ¿en qué medida las características logísticas, "
            "ambientales, económicas y geopolíticas de un envío permiten "
            "predecir la ocurrencia de una disrupción?"
        )
        st.markdown(
            "**Objetivo específico (indicador / técnica / proceso):** "
            "reducir la incertidumbre del **proceso** de planificación de "
            "riesgo en la cadena de suministro, estimando el **indicador** "
            "de probabilidad de disrupción de un envío mediante la **técnica** "
            "de Regresión Logística con regularización L1 y Random Forest, "
            "e integrando el resultado con herramientas de analítica "
            "(R y Power BI) usadas por el equipo de operaciones."
        )

    huapaya_datos_entrada(df)
    st.divider()
    modelos, _, escalador = huapaya_modelo(df)
    st.divider()
    c_logit = st.session_state.get("huapaya_c_logit", 0.5)
    n_arboles = st.session_state.get("huapaya_n_arboles", 200)
    huapaya_integracion(df, modelos, escalador, c_logit, n_arboles)


# =============================================================================
# PÁGINA PRINCIPAL / NAVEGACIÓN
# =============================================================================
def render_inicio():
    st.title("🚚 EC1 — Modelamiento Predictivo de Datos")
    st.markdown(
        "Aplicación conjunta de las cuatro dimensiones del proyecto sobre el dataset "
        "**Global Supply Chain Risk 2026**. Usa el menú de la barra lateral para navegar."
    )
    st.dataframe(pd.DataFrame({
        "Dimensión": ["CARRASCO", "REYES", "VILLANUEVA", "HUAPAYA"],
        "Tema": [
            "Proyección semanal de servicios",
            "Reparto multimodal: predicción del Lead Time",
            "Optimización de la cadena de abastecimiento",
            "Integración con Power BI y R",
        ],
        "Técnica": [
            "Holt (SES / lineal / amortiguado) + benchmarks",
            "Regresión lineal múltiple (OLS)",
            "XGBoost + SS y ROP",
            "Regresión logística L1 + Random Forest",
        ],
    }), use_container_width=True, hide_index=True)


SECCIONES = {
    "🏠 Inicio": None,
    "📈 1. CARRASCO — Proyección (Holt)": render_carrasco,
    "🚚 2. REYES — Reparto multimodal (Regresión lineal)": render_reyes,
    "📦 3. VILLANUEVA — Cadena de abastecimiento (XGBoost)": render_villanueva,
    "🧩 4. HUAPAYA — Power BI y R": render_huapaya,
}


def main():
    with st.sidebar:
        st.title("EC1 — Navegación")
        seleccion = st.radio("Sección", list(SECCIONES), key="nav_seccion")
        st.markdown("---")
        st.header("Carga de datos")
        archivo = st.file_uploader("Cargar dataset CSV (opcional)", type=["csv"], key="up_global")

    funcion = SECCIONES[seleccion]
    if funcion is None:
        render_inicio()
        return

    try:
        raw = cargar_raw(archivo)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    funcion(raw)


if __name__ == "__main__":
    main()
