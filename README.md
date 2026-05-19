# 🪄 Datathon Passos Mágicos - Fase 5

![Badge FIAP](https://img.shields.io/badge/FIAP-POSTECH-ed145b)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)

🔗 **Acesse o App Preditivo Online:** [fiap-tech-challenge-5-01.streamlit.app](https://fiap-tech-challenge-5-01.streamlit.app/)
💻 **Repositório do Projeto:** [github.com/mrcl-crz/fiap-tech-challenge-5](https://github.com/mrcl-crz/fiap-tech-challenge-5)

Projeto de análise de dados educacionais e desenvolvimento de um modelo preditivo de Machine Learning para a **Associação Passos Mágicos**, desenvolvido como Tech Challenge (Fase 5) da pós-graduação da FIAP.

## 🎯 O Desafio

A Associação Passos Mágicos transforma a vida de jovens de baixa renda por meio da educação. O objetivo deste projeto foi:
1. **Análise Exploratória:** Compreender a evolução do desempenho, engajamento e fatores psicossociais dos alunos nos anos de 2022 a 2024, respondendo a perguntas-chave de negócio.
2. **Modelo Preditivo:** Criar um modelo de inteligência artificial (Gradient Boosting) capaz de identificar alunos em **risco de defasagem acadêmica** usando apenas métricas e indicadores de processo contínuo (evitando *data leakage*).

## 🛠 Tecnologias Utilizadas
- **Linguagem:** Python
- **Análise e Machine Learning:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib
- **Deploy / Interface Preditiva:** Streamlit
- **Relatórios:** Apresentação em PDF

## 📁 Estrutura do Projeto

```text
├── apresentacao/          # Relatório executivo em PDF
├── modelo/                # Modelo treinado salvo em .joblib (Gradient Boosting + Scaler)
├── notebooks/
│   ├── analise_exploratoria.ipynb  # Jupyter Notebook com a Análise Exploratória (11 Perguntas)
│   ├── modelo_preditivo.ipynb      # Treino, avaliação e engenharia de variáveis do modelo
├── streamlit_app/
│   └── app.py             # Aplicação web final interativa (Deploy)
└── README.md
```

## 🚀 Como Executar

### 1. Aplicação Preditiva (Streamlit)
O modelo preditivo foi embarcado em uma aplicação interativa para uso da equipe pedagógica, permitindo calcular o risco educacional na ponta.

1. Instale as dependências essenciais:
   ```bash
   pip install streamlit pandas scikit-learn numpy joblib
   ```
2. Inicie o servidor do Streamlit:
   ```bash
   cd streamlit_app
   streamlit run app.py
   ```
3. A aplicação abrirá automaticamente no seu navegador padrão (`http://localhost:8501`).

### 2. Cadernos de Análise
Todos os achados estatísticos e o fluxo de modelagem estão detalhados na pasta `notebooks/`. Eles podem ser explorados utilizando qualquer ambiente compatível com Jupyter Notebooks (Jupyter Lab, VS Code, Google Colab).

## 👥 Equipe
- **Danielle Oliveira** (rm365922@fiap.com.br)
- **Marcelo Cruz de Sousa** (rm366336@fiap.com.br)
- **Monalisa Meyrelle Sousa** (rm366336@fiap.com.br)
- **Helio Ricardo Fortunato** (rm365341@fiap.com.br)