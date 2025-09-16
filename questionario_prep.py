import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sqlite3
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

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

# Inicialização de banco de dados
def init_db():
    conn = sqlite3.connect('prep_research.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS respostas
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  Conhecimento_PrEP TEXT,
                  Conhecimento_PEP TEXT,
                  Acesso_servicos TEXT,
                  Fonte_informacao TEXT,
                  Uso_PrepPEP TEXT,
                  Conhece_usuarios TEXT,
                  Teste_HIV_frequencia TEXT,
                  Metodos_prevencao TEXT,
                  Genero TEXT,
                  Orientacao_sexual TEXT,
                  Raca TEXT,
                  Faixa_etaria TEXT,
                  Renda TEXT,
                  Regiao TEXT)''')
    conn.commit()
    conn.close()

# Função para salvar dados no SQLite
def salvar_dados(resposta):
    # Inicializar banco de dados
    init_db()
    
    # Adicionar timestamp
    resposta['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Conectar ao banco de dados
    conn = sqlite3.connect('prep_research.db')
    c = conn.cursor()
    
    # Inserir dados
    c.execute('''INSERT INTO respostas 
                 (timestamp, Conhecimento_PrEP, Conhecimento_PEP, Acesso_servicos, 
                  Fonte_informacao, Uso_PrepPEP, Conhece_usuarios, Teste_HIV_frequencia,
                  Metodos_prevencao, Genero, Orientacao_sexual, Raca, Faixa_etaria, 
                  Renda, Regiao)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (resposta['timestamp'], resposta['Conhecimento_PrEP'], resposta['Conhecimento_PEP'],
               resposta['Acesso_servicos'], resposta['Fonte_informacao'], resposta['Uso_PrepPEP'],
               resposta['Conhece_usuarios'], resposta['Teste_HIV_frequencia'], resposta['Metodos_prevencao'],
               resposta['Genero'], resposta['Orientacao_sexual'], resposta['Raca'],
               resposta['Faixa_etaria'], resposta['Renda'], resposta['Regiao']))
    
    conn.commit()
    conn.close()
    
    # Atualizar dados na sessão
    if 'dados' not in st.session_state:
        st.session_state.dados = pd.DataFrame()
    
    novo_df = pd.DataFrame([resposta])
    st.session_state.dados = pd.concat([st.session_state.dados, novo_df], ignore_index=True)
    
    return True

# Carregar dados do banco
def carregar_dados():
    init_db()
    conn = sqlite3.connect('prep_research.db')
    df = pd.read_sql_query("SELECT * FROM respostas", conn)
    conn.close()
    
    # Remover coluna de ID se existir
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    return df

# Inicialização de dados
if 'dados' not in st.session_state:
    st.session_state.dados = carregar_dados()

# Barra lateral com menu de navegação
st.sidebar.image("https://img.icons8.com/color/96/000000/data-configuration.png", width=80)
    
# Menu de navegação usando componentes nativos do Streamlit
pagina = st.sidebar.radio(
    "Menu Principal",
    ["Questionário", "Visualizações", "Análises", "Sobre"]
)

# Informações sobre o projeto
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="info-box">
    <h4 style="color: #000000;">Pesquisa sobre Conhecimento de PrEP/PEP</h4>
    <p style="color: #000000;">Este projeto visa mapear o conhecimento sobre métodos de prevenção ao HIV na população de São Paulo.</p>
</div>
""", unsafe_allow_html=True)

# Página do Questionário
if pagina == "Questionário":
    # Cabeçalho
    st.markdown('<h1 class="main-header">Pesquisa sobre PrEP e Prevenção ao HIV em São Paulo</h1>', unsafe_allow_html=True)
    
    # Barra de progresso
    progresso = st.progress(0)
    st.markdown("**Progresso: 0% completado**")
    
    # Formulário de pesquisa
    with st.form("pesquisa_form"):
        # Parte 1: Conhecimento
        st.markdown('<h2 class="section-header">Parte 1/3: Conhecimento sobre PrEP/PEP</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            q1 = st.radio("**Você conhece a PrEP (Profilaxia Pré-Exposição)?**", [
                "Sim, conheço bem", 
                "Conheço parcialmente", 
                "Já ouvi falar mas não sei detalhes", 
                "Não conheço"
            ])
            
            # Pergunta de conhecimento sobre PrEP
            if q1 in ["Sim, conheço bem", "Conheço parcialmente"]:
                q1a = st.radio("**A PrEP é eficaz para prevenir qual destas ISTs?**", [
                    "Apenas HIV",
                    "HIV e sífilis",
                    "HIV e hepatite B",
                    "Todas as ISTs"
                ], help="Selecione a alternativa correta")
        
        with col2:
            q2 = st.radio("**E a PEP (Profilaxia Pós-Exposição)?**", [
                "Sim, conheço bem", 
                "Conheço parcialmente", 
                "Já ouvi falar mas não sei detalhes", 
                "Não conheço"
            ])
            
            # Pergunta de conhecimento sobre PEP
            if q2 in ["Sim, conheço bem", "Conheço parcialmente"]:
                q2a = st.radio("**Em quanto tempo após a exposição deve-se iniciar a PEP?**", [
                    "Até 24 horas",
                    "Até 72 horas",
                    "Até 1 semana",
                    "Não há prazo limite"
                ], help="Selecione a alternativa correta")
        
        # Atualizar progresso
        progresso.progress(33)
        st.markdown("**Progresso: 33% completado**")
        
        # Parte 2: Experiência Pessoal
        st.markdown('<h2 class="section-header">Parte 2/3: Experiência Pessoal</h2>', unsafe_allow_html=True)
        
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
                "Sim, já usei no passado",
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
        
        # Atualizar progresso
        progresso.progress(66)
        st.markdown("**Progresso: 66% completado**")
        
        # Parte 3: Perfil Demográfico
        st.markdown('<h2 class="section-header">Parte 3/3: Perfil Demográfico</h2>', unsafe_allow_html=True)
        
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
        
        # Questão aberta opcional
        st.markdown("---")
        comentarios = st.text_area("**Tem algum comentário ou sugestão sobre prevenção ao HIV?** (opcional)", 
                                 height=100,
                                 help="Seu comentário pode nos ajudar a melhorar os serviços de prevenção")
        
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
            
            # Adicionar perguntas de conhecimento se respondidas
            if 'q1a' in locals():
                resposta["Conhecimento_PrEP_Teste"] = q1a
            if 'q2a' in locals():
                resposta["Conhecimento_PEP_Teste"] = q2a
            if comentarios:
                resposta["Comentarios"] = comentarios
            
            if salvar_dados(resposta):
                progresso.progress(100)
                st.markdown("""
                <div class="success-box">
                    <h3 style="color: #000000;">✅ Obrigado por participar da pesquisa!</h3>
                    <p style="color: #000000;">Sua contribuição é muito importante para entendermos melhor o conhecimento sobre 
                    prevenção ao HIV em nossa comunidade.</p>
                    <p style="color: #000000;">Você pode visualizar os resultados preliminares na aba "Visualizações".</p>
                </div>
                """, unsafe_allow_html=True)
        elif enviado and not consentimento:
            st.error("Você precisa concordar com os termos de consentimento para enviar o formulário.")

# Página de Visualizações
elif pagina == "Visualizações":
    st.markdown('<h1 class="main-header">Visualizações dos Dados</h1>', unsafe_allow_html=True)
    
    if not st.session_state.dados.empty:
        # Estatísticas rápidas
        total_respostas = len(st.session_state.dados)
        st.markdown(f"""
        <div style="background-color: #E6F7FF; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem; color: #000000;">
            <h3 style="text-align: center; color: #000000;">📊 Total de Respostas: {total_respostas}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Filtros
        st.sidebar.header("Filtros")
        
        # Filtro por região
        regioes = st.sidebar.multiselect(
            "Filtrar por região:",
            options=st.session_state.dados['Regiao'].unique(),
            default=st.session_state.dados['Regiao'].unique()
        )
        
        # Aplicar filtros
        dados_filtrados = st.session_state.dados[st.session_state.dados['Regiao'].isin(regioes)]
        
        # Visualizações
        tab1, tab2, tab3 = st.tabs(["Distribuições", "Comparativos", "Mapas"])
        
        with tab1:
            # Gráfico de distribuição do conhecimento
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            
            conhecimento_prep = dados_filtrados['Conhecimento_PrEP'].value_counts()
            ax[0].bar(conhecimento_prep.index, conhecimento_prep.values)
            ax[0].set_title('Conhecimento sobre PrEP')
            ax[0].tick_params(axis='x', rotation=45)
            
            conhecimento_pep = dados_filtrados['Conhecimento_PEP'].value_counts()
            ax[1].bar(conhecimento_pep.index, conhecimento_pep.values)
            ax[1].set_title('Conhecimento sobre PEP')
            ax[1].tick_params(axis='x', rotation=45)
            
            st.pyplot(fig)
        
        with tab2:
            # Comparativo por gênero
            conhecimento_genero = pd.crosstab(
                dados_filtrados['Genero'], 
                dados_filtrados['Conhecimento_PrEP']
            )
            
            fig, ax = plt.subplots(figsize=(12, 6))
            conhecimento_genero.plot(kind='bar', ax=ax)
            ax.set_title('Conhecimento de PrEP por Identidade de Gênero')
            ax.legend(title="Conhecimento", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        
        with tab3:
            # Mapa de calor por região
            st.info("Em desenvolvimento: Mapa interativo das respostas por região")
    
    else:
        st.info("Aguardando respostas. As visualizações serão exibidas aqui quando houver dados suficientes.")

# Página de Análises
elif pagina == "Análises":
    st.markdown('<h1 class="main-header">Análises Avançadas</h1>', unsafe_allow_html=True)
    
    if not st.session_state.dados.empty and len(st.session_state.dados) >= 10:
        st.markdown("""
        <div class="ml-explanation">
            <h4 style="color: #000000;">🤖 Análise com Machine Learning</h4>
            <p style="color: #000000;">Utilizamos algoritmos de aprendizado de máquina para identificar padrões 
            e agrupamentos naturais nas respostas.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Análise de clusters
        try:
            dados_ml = st.session_state.dados.copy()
            
            # Codificar variáveis categóricas
            le = LabelEncoder()
            for col in dados_ml.select_dtypes(include=['object']).columns:
                if col != 'timestamp' and col != 'Metodos_prevencao' and col != 'Comentarios':
                    dados_ml[col] = le.fit_transform(dados_ml[col].astype(str))
            
            # Remover colunas problemáticas
            cols_remover = ['timestamp', 'Metodos_prevencao', 'Comentarios']
            dados_ml = dados_ml.drop([col for col in cols_remover if col in dados_ml.columns], axis=1)
            
            if len(dados_ml) >= 3:
                # Padronizar os dados
                scaler = StandardScaler()
                dados_scaled = scaler.fit_transform(dados_ml)
                
                # Aplicar K-Means
                n_clusters = min(3, len(dados_scaled) - 1)
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
                
                # Interpretação dos clusters
                st.info("""
                **Interpretação dos Clusters:** 
                - **Cluster 0**: Perfil com menor conhecimento e acesso
                - **Cluster 1**: Perfil com conhecimento intermediário
                - **Cluster 2**: Perfil com maior conhecimento e experiência
                """)
        
        except Exception as e:
            st.error(f"Erro na análise: {str(e)}")
    
    else:
        st.info("São necessárias pelo menos 10 respostas para as análises avançadas.")

# Página Sobre
elif pagina == "Sobre":
    st.markdown('<h1 class="main-header">Sobre o Projeto</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #000000;">Objetivos da Pesquisa</h4>
            <p style="color: #000000;">- Mapear o conhecimento sobre PrEP e PEP em São Paulo</p>
            <p style="color: #000000;">- Identificar barreiras de acesso aos serviços</p>
            <p style="color: #000000;">- Informar políticas públicas de prevenção ao HIV</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #000000;">Próximas Etapas</h4>
            <p style="color: #000000;">- Análise aprofundada dos resultados</p>
            <p style="color: #000000;">- Relatório com recomendações</p>
            <p style="color: #000000;">- Divulgação dos resultados para a comunidade</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #000000;">Metodologia</h4>
            <p style="color: #000000;">- Pesquisa quantitativa online</p>
            <p style="color: #000000;">- Amostra não probabilística</p>
            <p style="color: #000000;">- Análise estatística e machine learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #000000;">Contato</h4>
            <p style="color: #000000;">Para mais informações sobre a pesquisa:</p>
            <p style="color: #000000;">email@pesquisaprep.com</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #000000; margin-top: 2rem;">
        <p>Desenvolvido com Streamlit, Pandas e Scikit-learn | Pesquisa sobre Prevenção ao HIV</p>
    </div>
    """, unsafe_allow_html=True)

# Rodapé em todas as páginas
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #000000; margin-top: 2rem;">
    <p>© 2023 Pesquisa sobre PrEP e Prevenção ao HIV | Todos os dados são anônimos e confidenciais</p>
</div>
""", unsafe_allow_html=True)