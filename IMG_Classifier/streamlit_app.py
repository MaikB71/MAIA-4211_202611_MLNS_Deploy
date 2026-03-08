#  We ensure proper path handling in Python
import Definitions
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.ModelController import ModelController

### Setup and configuration

st.set_page_config(
    layout="centered", page_title="Clasificador de Textos ODS", page_icon="❄️"
)

### My vars

ctrl = ModelController()

### My UI starting here
st.header("Clasificador de Textos - Objetivos de Desarrollo Sostenible")
st.markdown("""
**📌 ¿Qué hace esta aplicación?**
- Carga archivos CSV con textos relacionados a ODS
- Utiliza un modelo de ML entrenado para clasificar automáticamente los textos
- Visualiza resultados y estadísticas de clasificación
""")

with st.form(key="my_form"):
    
    
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", accept_multiple_files=False, type="csv"
    )

    submit_button = st.form_submit_button(label="Submit")

if submit_button and uploaded_file is not None:
    input_df, is_valid = ctrl.load_input_data(uploaded_file)
    st.session_state["input_df"] = input_df if is_valid else None

input_df = st.session_state.get("input_df")

if input_df is not None:
    st.caption("✅ This is your data")
    event = st.dataframe(
        input_df,
        on_select="rerun",
        selection_mode="single-row",
        use_container_width=True,
    )
    st.caption("▶ Please select a row")

    if event is not None and event.selection.rows:        
        current_row_index = event.selection.rows[0]
        current_row = input_df.iloc[current_row_index]

        #TO-DO: Llama la clase de predicción para procesar la información
        X, Y, Y_pred,probabilidades = ctrl.predict(input_df)
        #TO-DO: Obten el nombre de las clases
        class_names = ctrl.get_categories()

        col1, col2 = st.columns([0.6, 0.4])  

        with col1:
            st.caption("🗣 Analyzed text")
            #TO-DO
            st.info(current_row['textos'])

        with col2:
            st.caption("🎯 Your results")
            #TO-DO
            st.metric("Real:"," ")
            st.success(f"ODS {Y[current_row_index]}")
            st.metric("Prediction:", " ")
            st.success(f"ODS {Y_pred[current_row_index]}")
            st.metric("Confianza", f"{max(probabilidades[current_row_index]):.2%}")
