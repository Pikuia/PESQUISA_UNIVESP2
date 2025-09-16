import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from streamlit_option_menu import option_menu

# Configuração da página
st.set_page_config(
    page_title="Pesquisa PrEP/HIV - São Paulo",
    page_icon="🏳️‍🌈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #4682B4;
        border-bottom: 2px solid #1E90FF;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1.5rem;
    }
    .stButton>button:hover {
        background-color: #4682B4;
    }
    .success-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #E6F7FF;
        border-left: 5px solid #1E90FF;
        margin-bottom: 1.5rem;
    }
    .info-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F0F8FF;
        border-left: 5px solid #4682B4;
        margin: 1rem 0;
    }
    .tech-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #F8F9FA;
        border-left: 5px solid #28A745;
        margin: 0.5rem 0;
    }
    .ml-explanation {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #FFFFFF;
        border: 2px solid #6F42C1;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Garantir contraste adequado */
    .stMarkdown, .stText, .stAlert, .stInfo, .stSuccess {
        color: #000000 !important;
    }
    div[data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }
    .stApp {
        background-color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Inicialização de dados
if 'dados' not in st.session_state:
    if os.path.exists("respostas_prep.csv"):
        st.session_state.dados = pd.read_csv("respostas_prep.csv")
    else:
        st.session_state.dados = pd.DataFrame()

# Função para salvar dados
def salvar_dados(resposta):
    arquivo_csv = "respostas_prep.csv"
    
    # Adicionar timestamp
    resposta['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if os.path.exists(arquivo_csv):
        df_existente = pd.read_csv(arquivo_csv)
        df_novo = pd.DataFrame([resposta])
        df_final = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        df_final = pd.DataFrame([resposta])
    
    df_final.to_csv(arquivo_csv, index=False)
    st.session_state.dados = df_final
    return True

# Barra lateral com menu de navegação
with st.sidebar:
    st.title("Sobre o Projeto")
    
    # Menu de navegação usando streamlit_option_menu
    selected = option_menu(
        menu_title="Menu Principal",
        options=["Questionário", "Visualizações", "Análises", "Sobre"],
        icons=["clipboard", "bar-chart", "cpu", "info-circle"],
        default_index=0,
    )
    
    # Informações sobre o projeto
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <h4 style="color: #000000;">Pesquisa sobre Conhecimento de PrEP/PEP</h4>
        <p style="color: #000000;">Este projeto visa mapear o conhecimento sobre métodos de prevenção ao HIV na população de São Paulo.</p>
    </div>
    """, unsafe_allow_html=True)

# Página do Questionário
if selected == "Questionário":
    # Cabeçalho
    st.markdown('<h1 class="main-header">Pesquisa sobre PrEP e Prevenção ao HIV em São Paulo</h1>', unsafe_allow_html=True)
    
    # Formulário de pesquisa
    with st.form("pesquisa_form"):
        # Parte 1: Conhecimento
        st.markdown('<h2 class="section-header">Parte 1: Conhecimento sobre PrEP/PEP</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            q1 = st.radio("**Você conhece a PrEP (Profilaxia Pré-Exposição)?**", [
                "Sim, conheço bem", 
                "Conheço parcialmente", 
                "Já ouvi falar mas não sei detalhes", 
                "Não conheço"
            ])
            
        with col2:
            q2 = st.radio("**E a PEP (Profilaxia Pós-Exposição)?**", [
                "Sim, conheço bem", 
                "Conheço parcialmente", 
                "Já ouvi falar mas não sei detalhes", 
                "Não conheço"
            ])
        
        # Parte 2: Experiência Pessoal
        st.markdown('<h2 class="section-header">Parte 2: Experiência Pessoal</h2>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            q3 = st.radio("**Você sabe onde conseguir PrEP/PEP em São Paulo?**", [
                "Sim, conheço vários serviços",
                "Conheço apenas um local",
                "Não sei mas gostaria de saber",
                "Não sei e não tenho interesse"
            ])
            
            q4 = st.radio("**Como você ficou sabendo sobre PrEP/PEP?**", [
                "Profissional de saúde",
                "Amigos/conhecidos",
                "Internet/redes sociais",
                "Material informativo (folhetos, cartazes)",
                "Nunca ouvi falar",
                "Outra fonte"
            ])
        
        with col4:
            q5 = st.radio("**Você já usou ou usa PrEP/PEP?**", [
                "Sim, uso atualmente",
                "Sim, já usei no pastado",
                "Não, mas pretendo usar",
                "Não uso e não tenho interesse",
                "Prefiro não responder"
            ])
            
            q6 = st.radio("**Conhece alguém que usa ou já usou PrEP/PEP?**", [
                "Sim, vários conhecidos",
                "Sim, algumas pessoas",
                "Não conheço ninguém",
                "Prefiro não responder"
            ])
            
            q7 = st.radio("**Com que frequência você faz teste de HIV?**", [
                "A cada 3 meses",
                "A cada 6 meses",
                "Uma vez por ano",
                "Raramente faço",
                "Nunca fiz",
                "Prefiro não responder"
            ])
        
        # Parte 3: Perfil Demográfico
        st.markdown('<h2 class="section-header">Parte 3: Perfil Demográfico</h2>', unsafe_allow_html=True)
        
        col5, col6 = st.columns(2)
        
        with col5:
            genero = st.selectbox("**Identidade de gênero:**", [
                "Mulher cisgênero",
                "Homem cisgênero",
                "Mulher trans/transgênero",
                "Homem trans/transgênero",
                "Pessoa não-binária",
                "Travesti",
                "Agênero",
                "Gênero fluido",
                "Outro",
                "Prefiro não responder"
            ])
            
            orientacao = st.selectbox("**Orientação sexual:**", [
                "Assexual",
                "Bissexual",
                "Gay",
                "Lésbica",
                "Pansexual",
                "Heterossexual",
                "Queer",
                "Outra",
                "Prefiro não responder"
            ])
            
            raca = st.radio("**Raça/Cor:**", [
                "Amarela (origem asiática)",
                "Branca",
                "Indígena",
                "Parda",
                "Preta",
                "Prefiro não responder"
            ])
        
        with col6:
            idade = st.radio("**Faixa etária:**", [
                "13-17", "18-24", "25-29", "30-39",
                "40-49", "50-59", "60+", "Prefiro não responder"
            ])
            
            renda = st.radio("**Renda mensal individual:**", [
                "Até 1 salário mínimo", 
                "1-2 salários mínimos",
                "2-3 salários mínimos", 
                "3-5 salários mínimos",
                "Mais de 5 salários mínimos",
                "Prefiro não responder"
            ])
            
            regiao = st.selectbox("**Região de São Paulo onde mora:**", [
                "Centro expandido",
                "Zona Norte",
                "Zona Sul",
                "Zona Leste",
                "Zona Oeste",
                "Região Metropolitana",
                "Não moro em São Paulo",
                "Prefiro não responder"
            ])
        
        # Métodos de prevenção
        q8 = st.multiselect("**Quais métodos de prevenção ao HIV você utiliza?**", [
            "PrEP",
            "PEP",
            "Camisinha masculina",
            "Camisinha feminina",
            "Testagem regular",
            "Não utilizo métodos de prevenção",
            "Outro"
        ])
        
        # Termos de consentimento
        st.markdown("""
        <div style="background-color: #F0F8FF; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; color: #000000;">
            <p><strong>Termo de Consentimento:</strong> Ao enviar este formulário, você concorda em participar desta pesquisa 
            e que seus dados anônimos sejam utilizados para fins de estudo estatístico. Todas as informações são confidenciais 
            e não serão compartilhadas de forma individual.</p>
        </div>
        """, unsafe_allow_html=True)
        
        consentimento = st.checkbox("**Eu concordo em participar da pesquisa**", value=False)
        
        # Botão de envio
        enviado = st.form_submit_button("Enviar Respostas")
        
        if enviado and consentimento:
            resposta = {
                "Conhecimento_PrEP": q1,
                "Conhecimento_PEP": q2,
                "Acesso_servicos": q3,
                "Fonte_informacao": q4,
                "Uso_PrepPEP": q5,
                "Conhece_usuarios": q6,
                "Teste_HIV_frequencia": q7,
                "Metodos_prevencao": ", ".join(q8),
                "Genero": genero,
                "Orientacao_sexual": orientacao,
                "Raca": raca,
                "Faixa_etaria": idade,
                "Renda": renda,
                "Regiao": regiao
            }
            
            if salvar_dados(resposta):
                st.markdown("""
                <div class="success-box">
                    <h3 style="color: #000000;">✅ Obrigado por participar da pesquisa!</h3>
                    <p style="color: #000000;">Sua contribuição é muito importante para entendermos melhor o conhecimento sobre 
                    prevenção ao HIV em nossa comunidade.</p>
                </div>
                """, unsafe_allow_html=True)
        elif enviado and not consentimento:
            st.error("Você precisa concordar com os termos de consentimento para enviar o formulário.")

# Página de Visualizações
elif selected == "Visualizações":
    st.markdown('<h1 class="main-header">Visualizações dos Dados</h1>', unsafe_allow_html=True)
    
    if not st.session_state.dados.empty:
        # Estatísticas rápidas
        total_respostas = len(st.session_state.dados)
        st.markdown(f"""
        <div style="background-color: #E6F7FF; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem; color: #000000;">
            <h3 style="text-align: center; color: #000000;">📊 Total de Respostas: {total_respostas}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Gráfico de distribuição do conhecimento
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        conhecimento_prep = st.session_state.dados['Conhecimento_PrEP'].value_counts()
        ax[0].bar(conhecimento_prep.index, conhecimento_prep.values)
        ax[0].set_title('Conhecimento sobre PrEP')
        ax[0].tick_params(axis='x', rotation=45)
        
        conhecimento_pep = st.session_state.dados['Conhecimento_PEP'].value_counts()
        ax[1].bar(conhecimento_pep.index, conhecimento_pep.values)
        ax[1].set_title('Conhecimento sobre PEP')
        ax[1].tick_params(axis='x', rotation=45)
        
        st.pyplot(fig)
    
    else:
        st.info("Aguardando respostas. As visualizações serão exibidas aqui quando houver dados suficientes.")

# Página de Análises
elif selected == "Análises":
    st.markdown('<h1 class="main-header">Análises Avançadas</h1>', unsafe_allow_html=True)
    
    if not st.session_state.dados.empty and len(st.session_state.dados) >= 3:
        try:
            # Preparar dados para clustering
            dados_ml = st.session_state.dados.copy()
            
            # Codificar variáveis categóricas
            le = LabelEncoder()
            for col in dados_ml.select_dtypes(include=['object']).columns:
                dados_ml[col] = le.fit_transform(dados_ml[col].astype(str))
            
            # Padronizar os dados
            scaler = StandardScaler()
            dados_scaled = scaler.fit_transform(dados_ml)
            
            # Aplicar K-Means
            n_clusters = min(3, len(dados_scaled))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(dados_scaled)
            
            # Reduzir dimensionalidade para visualização
            pca = PCA(n_components=2)
            componentes = pca.fit_transform(dados_scaled)
            
            # Visualizar clusters
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(componentes[:, 0], componentes[:, 1], c=clusters, cmap='viridis', alpha=0.7)
            ax.set_xlabel('Componente Principal 1')
            ax.set_ylabel('Componente Principal 2')
            ax.set_title('Agrupamento de Respostas (K-Means)')
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Erro na análise: {str(e)}")
    
    else:
        st.info("São necessárias pelo menos 3 respostas para as análises avançadas.")

# Página Sobre
elif selected == "Sobre":
    st.markdown('<h1 class="main-header">Sobre o Projeto</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <h4 style="color: #000000;">Objetivo da Pesquisa</h4>
        <p style="color: #000000;">Esta pesquisa visa mapear o conhecimento sobre PrEP e PEP na população de São Paulo, 
        identificando lacunas de informação e barreiras de acesso.</p>
    </div>
    """, unsafe_allow_html=True)

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #000000; margin-top: 2rem;">
    <p>Desenvolvido com Streamlit | Pesquisa sobre Prevenção ao HIV</p>
</div>
""", unsafe_allow_html=True)