import gradio as gr
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# Base de datos utilizada por el prototipo SIPROT
DATA_FILE = Path(__file__).parent / "dataset_diagnostico_ia.xlsx"
SHEET_NAME = "BD_trulas_pedidos_dataset"

pedidos = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)

# Preparar variables derivadas y objetivo, siguiendo el cuaderno original
pedidos["Fecha_Pedido"] = pd.to_datetime(pedidos["Fecha_Pedido"])
pedidos["Fecha_Entrega_Pactada"] = pd.to_datetime(
    pedidos["Fecha_Entrega_Pactada"]
)
pedidos["Plazo_Pactado_Dias"] = (
    pedidos["Fecha_Entrega_Pactada"] - pedidos["Fecha_Pedido"]
).dt.days
pedidos["Presento_Retraso"] = (pedidos["Dias_Retraso"] > 0).astype(int)

X = pedidos[
    ["Tipo_Prenda", "Cantidad_Unidades", "Canal_Venta", "Plazo_Pactado_Dias"]
]
y = pedidos["Presento_Retraso"]

preparador_app = ColumnTransformer([
    (
        "categoricas",
        OneHotEncoder(handle_unknown="ignore"),
        ["Tipo_Prenda", "Canal_Venta"],
    ),
    (
        "numericas",
        StandardScaler(),
        ["Cantidad_Unidades", "Plazo_Pactado_Dias"],
    ),
])

modelo_app = Pipeline([
    ("preparacion", preparador_app),
    ("modelo", GradientBoostingClassifier(random_state=42)),
])
modelo_app.fit(X, y)

def analizar_pedido(tipo_prenda, cantidad, canal, plazo):
    if tipo_prenda is None or canal is None:
        return "Selecciona el tipo de prenda y el canal de venta."
    if cantidad is None or plazo is None:
        return "Ingresa la cantidad de unidades y los días pactados."
    if cantidad <= 0 or plazo <= 0:
        return "La cantidad de unidades y los días pactados deben ser mayores que cero."

    nuevo_pedido = pd.DataFrame([{
        "Tipo_Prenda": tipo_prenda,
        "Cantidad_Unidades": cantidad,
        "Canal_Venta": canal,
        "Plazo_Pactado_Dias": plazo,
    }])

    prediccion = modelo_app.predict(nuevo_pedido)[0]
    probabilidades = modelo_app.predict_proba(nuevo_pedido)[0]
    probabilidad_retraso = probabilidades[1] * 100

    if prediccion == 1:
        resultado = "CLASIFICACIÓN: POSIBLE RETRASO"
        recomendacion = (
            "Revisar la programación de producción, la disponibilidad "
            "de insumos y las tareas pendientes."
        )
    else:
        resultado = "CLASIFICACIÓN: SIN RETRASO IDENTIFICADO"
        recomendacion = (
            "Mantener el seguimiento habitual del pedido y verificar "
            "el avance de producción."
        )

    return (
        f"{resultado}\n\n"
        f"Probabilidad estimada por el modelo: {probabilidad_retraso:.1f}%\n\n"
        f"Recomendación: {recomendacion}\n\n"
        "IMPORTANTE: Resultado experimental. El modelo no está validado "
        "para decisiones reales de producción."
    )

aplicacion = gr.Interface(
    fn=analizar_pedido,
    inputs=[
        gr.Dropdown(
            choices=sorted(pedidos["Tipo_Prenda"].dropna().unique().tolist()),
            label="Tipo de prenda",
        ),
        gr.Number(label="Cantidad de unidades", value=100, precision=0),
        gr.Dropdown(
            choices=sorted(pedidos["Canal_Venta"].dropna().unique().tolist()),
            label="Canal de venta",
        ),
        gr.Number(
            label="Días pactados para la entrega",
            value=20,
            precision=0,
        ),
    ],
    outputs=gr.Textbox(label="Resultado del análisis de SIPROT", lines=8),
    title="SIPROT - Inteligencia Artificial",
    description=(
        "Sistema experimental de análisis de pedidos de TRULAS S.A.S. "
        "Registra los datos y consulta la clasificación generada por el modelo."
    ),
)

if __name__ == "__main__":
    aplicacion.launch()
