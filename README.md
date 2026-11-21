# EC1 — Modelamiento Predictivo de Datos

Aplicación web en **Streamlit** que reúne las cuatro dimensiones del proyecto en una sola página, con una sección independiente para cada una. Todas trabajan sobre el dataset **Global Supply Chain Risk 2026**.

## Secciones

| # | Dimensión | Tema | Técnica |
|---|-----------|------|---------|
| 1 | **CARRASCO** | Proyección semanal de servicios | Holt (SES / lineal / amortiguado) con selección por validación de origen móvil, benchmarks, Ridge y SVR, e intervalos de predicción |
| 2 | **REYES** | Reparto multimodal: predicción del Lead Time | Regresión lineal múltiple (OLS), PCA, métricas RMSE/MAE/R² y resumen real vs. predicho por modo de transporte |
| 3 | **VILLANUEVA** | Optimización de la cadena de abastecimiento | Clasificación de disrupción con XGBoost, Stock de Seguridad (SS) y Punto de Reorden (ROP), PCA vs LDA vs ANOVA F |
| 4 | **HUAPAYA** | Integración con Power BI y R | Regresión logística L1 y Random Forest, PCA, exportación de CSV y script R |

La navegación se hace desde la barra lateral. Solo se ejecuta la sección seleccionada, lo que mantiene la app rápida.

## Estructura del repositorio

```
.
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── global_supply_chain_risk_2026.csv
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

La app busca el CSV, en este orden, en:

1. `data/global_supply_chain_risk_2026.csv`
2. `global_supply_chain_risk_2026.csv`
3. `modelamiento.csv`

También puedes subirlo desde la barra lateral (**Cargar dataset CSV**). Si el archivo no se sube al repositorio, la app pedirá cargarlo manualmente.

Columnas esperadas: `Shipment_ID`, `Date`, `Origin_Port`, `Destination_Port`, `Transport_Mode`, `Product_Category`, `Weather_Condition`, `Distance_km`, `Weight_MT`, `Fuel_Price_Index`, `Geopolitical_Risk_Score`, `Carrier_Reliability_Score`, `Lead_Time_Days`, `Disruption_Occurred`.

## Despliegue en Streamlit Community Cloud

1. Sube el repositorio a GitHub (incluye `app.py`, `requirements.txt` y, si quieres, el CSV en `data/`).
2. Entra a [share.streamlit.io](https://share.streamlit.io), crea una app nueva y selecciona el repositorio.
3. Indica `app.py` como archivo principal.

## Notas

- Cada dimensión usa su propia copia del dataframe y claves de widgets con prefijo propio (`c_`, `v_`, `huapaya_`), por lo que no interfieren entre sí.
- Las implementaciones equivalentes en R (regresión lineal, Holt, logística L1 y Random Forest) se muestran y pueden descargarse dentro de cada sección.
- El script R de la dimensión REYES (regresión lineal para R Visual de Power BI) se muestra dentro de la sección y puede copiarse directamente.
