# ======================
# IMPORTACIÓN DE LIBRERÍAS
# ======================
import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# CONFIGURACIÓN INICIAL
# ======================
st.set_page_config(
    page_title="Sophivigil Dashboard",
    layout="wide",
    page_icon="👁️",
    initial_sidebar_state="expanded"
)

# ======================
# FUNCIONES AUXILIARES
# ======================
@st.cache_resource(show_spinner=False)
def get_faker_instance():
    return Faker('es_MX')

@st.cache_data(show_spinner="Generando datos sintéticos...")
def generate_adr_data(num_records=15):
    fake = get_faker_instance()
    productos = [
        "3-A Ofteno", "Acquafil Ofteno", "Deltamid", "Dustalox", "Eliptic Ofteno",
        "Gaap Ofteno", "Humylub Ofteno", "Krytantek Ofteno", "Lagricel Ofteno",
        "Manzanilla Sophia", "Meticel Ofteno", "Nazil", "Sopixín DX", "Trazidex"
    ]
    
    relaciones = ["Paciente", "Familiar", "Profesional de salud"]
    severidades = ["Leve", "Moderado", "Grave"]
    data = []
    hoy = datetime.now().date()
    
    for _ in range(num_records):
        # Datos demográficos
        dob = fake.date_of_birth(minimum_age=20, maximum_age=80)
        genero = random.choice(["M", "F"])
        
        # Tratamiento
        start_treatment = fake.date_this_year(before_today=True, after_today=False)
        continues_treatment = random.choice([True, False])
        end_treatment = fake.date_between(start_date=start_treatment, end_date=hoy) if not continues_treatment else None
        
        # Iniciales consistentes
        nombre = fake.first_name()
        apellido = fake.last_name()
        iniciales = f"{nombre[0]}{apellido[0]}"
        if ' ' in nombre:
            iniciales = f"{nombre[0]}{nombre.split(' ')[1][0]}{apellido[0]}"
        
        # Reacción adversa
        prob_antes_tratamiento = 0.3
        if random.random() < prob_antes_tratamiento:
            fecha_min = start_treatment - timedelta(days=30)
            fecha_max = start_treatment
        else:
            fecha_min = start_treatment
            fecha_max = min(start_treatment + timedelta(days=90), hoy)
        
        onset = fake.date_between_dates(date_start=fecha_min, date_end=fecha_max)
        
        reaccion_continua = random.choice([True, False])
        end_reaction = fake.date_between_dates(date_start=onset, date_end=hoy) if not reaccion_continua else None

        # Descripción con términos médicos realistas
        sintomas = ["enrojecimiento", "picazón", "visión borrosa", "dolor ocular", 
                   "sequedad", "sensibilidad a la luz", "inflamación", "lagrimeo"]
        descripcion = f"Paciente reporta {random.choice(sintomas)}"
        if random.random() > 0.5:
            descripcion += f" acompañado de {random.choice(sintomas)}"
        descripcion += f". {fake.sentence()}"
        
        record = {
            "ID": fake.unique.bothify(text='RPT-#####'),
            "Iniciales": iniciales,
            "Edad": hoy.year - dob.year,
            "Género": genero,
            "País": fake.country_code(representation="alpha-2"),
            "Producto": random.choice(productos),
            "Inicio_Tratamiento": start_treatment,
            "Duración_Tratamiento": (end_treatment - start_treatment).days if end_treatment else (hoy - start_treatment).days,
            "Continúa": "Si" if continues_treatment else "No",
            "Fin_Tratamiento": end_treatment,
            "Lote": fake.bothify(text='LOT-??###'),
            "Evento_Adverso": fake.sentence(nb_words=4),
            "Inicio_Reacción": onset,
            "Días_Inicio_Reacción": (onset - start_treatment).days,
            "Fin_Reacción": end_reaction,
            "Duración_Reacción": (end_reaction - onset).days if end_reaction else (hoy - onset).days,
            "Descripción": descripcion,
            "Reportero": fake.name(),
            "Relación": random.choice(relaciones),
            "Severidad_Reportada": random.choice(severidades),
            "Teléfono": fake.phone_number(),
            "Email": fake.email(),
            "Fecha_Reporte": fake.date_between(start_date=onset, end_date=hoy)
        }
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # Convertir a formato datetime
    date_cols = ["Inicio_Tratamiento", "Fin_Tratamiento", "Inicio_Reacción", 
                "Fin_Reacción", "Fecha_Reporte"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df

# ======================
# INTERFAZ PRINCIPAL
# ======================
# Título y separador
st.title("👁️ Sophivigil – Sistema de Farmacovigilancia Oftálmica")
st.markdown("---")

# ======================
# BARRA LATERAL (CONTROLES)
# ======================
with st.sidebar:
    st.header("⚙️ Configuración")
    num_records = st.slider("Número de reportes", 5, 50, 15)
    st.markdown("---")
    
    st.header("🔍 Filtros")
    all_products = st.checkbox("Todos los productos", True, key="all_products")
    severidad_filtro = st.multiselect("Nivel de severidad:", options=["Leve", "Moderado", "Grave"])
    fecha_filtro = st.date_input("Reportes desde:", value=datetime.now() - timedelta(days=180))
    st.markdown("---")
    
    st.info("Sophivigil v1.0 | Sistema de monitoreo de eventos adversos oftálmicos")

# ======================
# GENERACIÓN DE DATOS
# ======================
df = generate_adr_data(num_records)

# ======================
# SIMULACIÓN DE PREDICCIONES
# ======================
# Generar predicciones ANTES del filtrado
np.random.seed(42)
probabilidades = np.random.dirichlet([1, 1, 1], size=len(df))
pred_labels = np.random.choice(["Leve", "Moderado", "Grave"], size=len(df), p=[0.5, 0.3, 0.2])
df["Predicción"] = pred_labels
df["Confianza"] = [f"{p.max()*100:.1f}%" for p in probabilidades]

# Actualizar opciones de filtro en sidebar
if not st.session_state.all_products:
    with st.sidebar:
        productos_filtro = st.multiselect(
            "Productos específicos:", 
            options=df["Producto"].unique(),
            default=[]
        )
else:
    productos_filtro = []

# ======================
# FILTRADO DE DATOS
# ======================
filtered_df = df.copy()

# Aplicar filtros
if not st.session_state.all_products and productos_filtro:
    filtered_df = filtered_df[filtered_df["Producto"].isin(productos_filtro)]
    
if severidad_filtro:
    filtered_df = filtered_df[filtered_df["Predicción"].isin(severidad_filtro)]
    
filtered_df = filtered_df[filtered_df["Fecha_Reporte"] >= pd.to_datetime(fecha_filtro)]

# ======================
# SECCIÓN 1: KPI Y ALERTAS
# ======================
st.header("📊 Resumen General")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Reportes", len(filtered_df), delta=f"{len(filtered_df)-len(df)} desde última actualización")
with col2:
    graves = filtered_df[filtered_df["Predicción"] == "Grave"]
    st.metric("Casos Graves", len(graves), delta_color="inverse")
with col3:
    productos_count = filtered_df["Producto"].nunique()
    st.metric("Productos Reportados", productos_count)
with col4:
    avg_response = filtered_df["Días_Inicio_Reacción"].mean() if not filtered_df.empty else 0
    st.metric("Tiempo Reacción Promedio", f"{avg_response:.1f} días")

# Alertas para casos graves
if not graves.empty:
    st.error(f"🚨 **ALERTA:** Se detectaron {len(graves)} casos graves que requieren atención inmediata!")
    with st.expander("Ver detalles de casos graves"):
        st.dataframe(graves[["ID", "Producto", "Evento_Adverso", "Inicio_Reacción", "Descripción", "Confianza"]])

st.markdown("---")

# ======================
# SECCIÓN 2: VISUALIZACIONES (TABS)
# ======================
st.header("📈 Análisis Visual")
tab1, tab2, tab3, tab4 = st.tabs(["Distribución", "Tendencias", "Texto", "Mapa"])

# Tab 1: Distribución
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Severidad de Eventos")
        if not filtered_df.empty:
            fig = px.pie(
                filtered_df, 
                names="Predicción", 
                color="Predicción",
                color_discrete_map={"Leve":"#2ecc71", "Moderado":"#f39c12", "Grave":"#e74c3c"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos para mostrar")
    
    with col2:
        st.subheader("Top Productos Reportados")
        if not filtered_df.empty:
            top_products = filtered_df["Producto"].value_counts().nlargest(5)
            fig = px.bar(
                top_products, 
                orientation="h",
                labels={'value':'Reportes', 'index':'Producto'},
                color=top_products.values,
                color_continuous_scale="Bluered"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos para mostrar")

# Tab 2: Tendencias
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Eventos por Mes")
        if not filtered_df.empty:
            df_mensual = filtered_df.set_index("Fecha_Reporte").resample('M').size()
            fig = px.line(
                df_mensual, 
                labels={'value':'Reportes', 'index':'Fecha'},
                title="Tendencia Mensual de Reportes"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos para mostrar")
    
    with col2:
        st.subheader("Duración de Reacciones")
        if not filtered_df.empty:
            fig = px.box(
                filtered_df, 
                y="Duración_Reacción", 
                x="Predicción",
                color="Predicción",
                points="all"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos para mostrar")

# Tab 3: Análisis de texto
with tab3:
    st.subheader("Análisis de Texto")
    
    if not filtered_df.empty:
        # Nube de palabras
        st.write("**Palabras más frecuentes en descripciones**")
        text = " ".join(filtered_df["Descripción"].tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
        
        # TF-IDF
        st.write("**Términos más relevantes (TF-IDF)**")
        tfidf = TfidfVectorizer(max_features=10, stop_words=['de', 'la', 'el', 'en', 'y', 'que'])
        X = tfidf.fit_transform(filtered_df["Descripción"])
        tfidf_df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())
        st.bar_chart(tfidf_df.mean().sort_values(ascending=False))
    else:
        st.warning("No hay datos para mostrar")

# Tab 4: Mapa
with tab4:
    st.subheader("Distribución Geográfica")
    if not filtered_df.empty:
        country_counts = filtered_df["País"].value_counts().reset_index()
        country_counts.columns = ['País', 'Reportes']
        
        fig = px.choropleth(
            country_counts,
            locations="País",
            color="Reportes",
            hover_name="País",
            projection="natural earth",
            color_continuous_scale="Viridis",
            title="Reportes por País"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos para mostrar")

st.markdown("---")

# ======================
# SECCIÓN 3: DATOS DETALLADOS
# ======================
st.header("📋 Datos de Reportes")
with st.expander("Ver datos completos", expanded=False):
    if not filtered_df.empty:
        st.dataframe(filtered_df.style.background_gradient(
            subset=["Duración_Reacción", "Duración_Tratamiento"], 
            cmap="YlOrRd"
        ))
    else:
        st.warning("No hay datos disponibles con los filtros actuales")

# ======================
# SECCIÓN 4: PLAN DE IMPLEMENTACIÓN
# ======================
st.header("🚀 Próximos Pasos")
with st.expander("Plan de implementación", expanded=False):
    st.markdown("""
    **Siguientes pasos para producción**:
    
    1. **Modelo Predictivo Real**  
       - Reemplazar modelo simulado con RandomForest/XGBoost entrenado
       - Incorporar embeddings clínicos (BioBERT)
    
    2. **Integración de Datos**  
       - Conectar con base de datos PostgreSQL/MySQL
       - API para recepción de reportes en tiempo real
    
    3. **Sistema de Alertas**  
       - Notificaciones push a equipos médicos
       - Integración con Slack/Teams/Correo
    
    4. **Monitoreo Continuo**  
       - Dashboard de performance del modelo
       - Sistema de retraining automático
    
    5. **Funcionalidades Adicionales**  
       - Búsqueda de casos similares
       - Análisis de señales de seguridad
       - Integración con sistemas de EHR
    """)
    
    st.progress(0.35, text="Estado actual del proyecto")

# ======================
# FOOTER
# ======================
st.markdown("---")
st.caption("© 2023 Sophivigil - Sistema de Farmacovigilancia Oftálmica | Datos simulados con propósitos demostrativos")