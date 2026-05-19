import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Passos Mágicos - Predição de Risco",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Passos Mágicos - Predição de Risco de Defasagem")
st.markdown(
    "Aplicação para prever o risco de defasagem educacional dos alunos "
    "com base em indicadores de processo e perfil — **sem uso de variáveis de resultado**."
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'modelo')

@st.cache_resource
def load_model():
    modelo   = joblib.load(os.path.join(MODEL_DIR, 'modelo_risco.joblib'))
    scaler   = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    features = joblib.load(os.path.join(MODEL_DIR, 'features.joblib'))
    return modelo, scaler, features

try:
    modelo, scaler, features = load_model()
    model_loaded = True
except Exception:
    model_loaded = False
    st.warning("⚠️ Modelo não encontrado. Execute o notebook `modelo_preditivo` primeiro.")

tab1, tab2 = st.tabs(["📊 Predição Individual", "ℹ️ Sobre"])

with tab1:
    if model_loaded:
        st.subheader("Insira os dados do aluno")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Indicadores de Processo**")
            iaa = st.number_input("IAA (Autoavaliação)",    0.0, 10.0, 8.0, 0.1)
            ieg = st.number_input("IEG (Engajamento)",      0.0, 10.0, 7.5, 0.1)
            ips = st.number_input("IPS (Psicossocial)",     0.0, 10.0, 6.0, 0.1)
            ipp = st.number_input("IPP (Psicopedagógico)",  0.0, 10.0, 7.0, 0.1)
            ida = st.number_input("IDA (Desempenho Acad.)", 0.0, 10.0, 6.5, 0.1)
            ipv = st.number_input("IPV (Ponto de Virada)",  0.0, 10.0, 7.0, 0.1)

        with col2:
            st.markdown("**Notas das Disciplinas**")
            mat = st.number_input("Nota Matemática",  0.0, 10.0, 6.0, 0.1)
            por = st.number_input("Nota Português",   0.0, 10.0, 6.5, 0.1)
            ing = st.number_input("Nota Inglês",      0.0, 10.0, 6.0, 0.1)

        with col3:
            st.markdown("**Perfil do Aluno**")
            idade  = st.number_input("Idade", 6, 30, 12)
            genero = st.selectbox("Gênero", ["Feminino", "Masculino"])
            tempo  = st.number_input("Anos na Instituição", 0, 10, 2)
            fase   = st.selectbox("Fase", [
                "ALFA","FASE 1","FASE 2","FASE 3","FASE 4",
                "FASE 5","FASE 6","FASE 7","FASE 8"
            ])

        fase_map = {
            'ALFA':0,'FASE 1':1,'FASE 2':2,'FASE 3':3,'FASE 4':4,
            'FASE 5':5,'FASE 6':6,'FASE 7':7,'FASE 8':8
        }

        if st.button("🔍 Prever Risco", type="primary"):
            input_data = pd.DataFrame([{
                'IAA': iaa, 'IEG': ieg, 'IPS': ips, 'IPP': ipp,
                'IDA': ida, 'IPV': ipv,
                'Mat': mat, 'Por': por, 'Ing': ing,
                'Idade': idade,
                'Genero_enc': 0 if genero == 'Feminino' else 1,
                'Tempo_inst': tempo,
                'Fase_enc': fase_map[fase]
            }])
            input_data   = input_data[features]
            input_scaled = scaler.transform(input_data)

            prob = modelo.predict_proba(input_scaled)[0]
            pred = modelo.predict(input_scaled)[0]

            st.markdown("---")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if pred == 1:
                    st.error("⚠️ **ALUNO EM RISCO DE DEFASAGEM**")
                else:
                    st.success("✅ **Aluno SEM risco de defasagem**")
            with col_r2:
                st.metric("Probabilidade de Risco", f"{prob[1]*100:.1f}%")
                st.progress(float(prob[1]))

with tab2:
    st.subheader("Sobre o Projeto")
    st.markdown("""
    **Datathon Fase 5 - POSTECH FIAP**

    Este projeto analisa os dados da Associação Passos Mágicos (2022–2024)
    para prever o risco de defasagem educacional dos alunos.

    **Modelo:** Gradient Boosting

    **Indicadores de processo utilizados:**
    - **IAA**: Indicador de Autoavaliação
    - **IEG**: Indicador de Engajamento
    - **IPS**: Indicador Psicossocial
    - **IPP**: Indicador Psicopedagógico
    - **IDA**: Indicador de Desempenho Acadêmico
    - **IPV**: Indicador de Ponto de Virada

    **Notas das disciplinas:** Matemática, Português, Inglês

    **Perfil:** Idade, Gênero, Tempo na instituição, Fase

    ---
    > **Nota técnica:** As variáveis IAN, Pedra e INDE foram excluídas do modelo
    > pois são derivadas do resultado final de avaliação do aluno (Defasagem),
    > o que causaria *data leakage* e resultados artificialmente perfeitos.
    """)
