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

def create_demographics(fake):
    dob = fake.date_of_birth(minimum_age=20, maximum_age=80)
    genero = random.choice(["M", "F"])
    nombre = fake.first_name()
    apellido = fake.last_name()
    
    # Generate initials safely
    parts = [p.strip() for p in nombre.split() if p.strip()]
    
    if len(parts) >= 2:
        # Handle potential single-character parts
        first_char = parts[0][0] if parts[0] else ''
        second_char = parts[1][0] if len(parts[1]) > 0 else ''
        iniciales = f"{first_char}{second_char}{apellido[0]}"
    else:
        # Fallback to first char of first name + last name
        iniciales = f"{nombre[0]}{apellido[0]}" if nombre and apellido else "XX"
    
    return dob, genero, iniciales

def create_treatment_data(fake, hoy):
    start_treatment = fake.date_this_year(before_today=True, after_today=False)
    continues_treatment = random.choice([True, False])
    end_treatment = fake.date_between(start_date=start_treatment, end_date=hoy) if not continues_treatment else None
    return start_treatment, continues_treatment, end_treatment

def create_reaction_data(fake, start_treatment, hoy):
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
    return onset, reaccion_continua, end_reaction

def create_description(fake):
    sintomas = ["enrojecimiento", "picazón", "visión borrosa", "dolor ocular", 
               "sequedad", "sensibilidad a la luz", "inflamación", "lagrimeo"]
    descripcion = f"Paciente reporta {random.choice(sintomas)}"
    if random.random() > 0.5:
        descripcion += f" acompañado de {random.choice(sintomas)}"
    descripcion += f". {fake.sentence()}"
    return descripcion

@st.cache_data(show_spinner="Generando datos sintéticos...")
def generate_adr_data(num_records=15):
    # Validate minimum records
    if num_records < 5:
        num_records = 5
        st.warning("Número mínimo de registros ajustado a 5 para análisis significativo")
    
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
        # Datos demográficos con validación
        dob, genero, iniciales = create_demographics(fake)
        
        # Tratamiento
        start_treatment, continues_treatment, end_treatment = create_treatment_data(fake, hoy)
        
        # Reacción adversa
        onset, reaccion_continua, end_reaction = create_reaction_data(fake, start_treatment, hoy)
        
        # Descripción
        descripcion = create_description(fake)
        
        # Cálculo de duraciones con validación de fechas
        try:
            duracion_tratamiento = (end_treatment - start_treatment).days if end_treatment else (hoy - start_treatment).days
            duracion_reaccion = (end_reaction - onset).days if end_reaction else (hoy - onset).days
        except TypeError:
            # Fallback if date calculation fails
            duracion_tratamiento = random.randint(30, 365)
            duracion_reaccion = random.randint(1, 90)

        record = {
            "ID": fake.unique.bothify(text='RPT-#####'),
            "Iniciales": iniciales,
            "Edad": hoy.year - dob.year,
            "Género": genero,
            "País": fake.country_code(representation="alpha-2"),
            "Producto": random.choice(productos),
            "Inicio_Tratamiento": start_treatment,
            "Duración_Tratamiento": duracion_tratamiento,
            "Continúa": "Si" if continues_treatment else "No",
            "Fin_Tratamiento": end_treatment,
            "Lote": fake.bothify(text='LOT-??###'),
            "Evento_Adverso": fake.sentence(nb_words=4),
            "Inicio_Reacción": onset,
            "Días_Inicio_Reacción": (onset - start_treatment).days,
            "Fin_Reacción": end_reaction,
            "Duración_Reacción": duracion_reaccion,
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
    
    # Convertir a formato datetime con manejo de errores
    date_cols = ["Inicio_Tratamiento", "Fin_Tratamiento", "Inicio_Reacción", 
                "Fin_Reacción", "Fecha_Reporte"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Fill missing dates with reasonable values
            if df[col].isnull().any():
                df[col] = df[col].fillna(pd.Timestamp(datetime.now().date()))
    
    return df

def apply_filters(df, all_products, productos_filtro, severidad_filtro, fecha_filtro):
    filtered_df = df.copy()
    
    if not all_products and productos_filtro:
        filtered_df = filtered_df[filtered_df["Producto"].isin(productos_filtro)]
        
    if severidad_filtro:
        filtered_df = filtered_df[filtered_df["Predicción"].isin(severidad_filtro)]
    
    if fecha_filtro:
        filtered_df = filtered_df[filtered_df["Fecha_Reporte"] >= pd.to_datetime(fecha_filtro)]
    
    return filtered_df

def generate_predictions(df):
    np.random.seed(42)
    pred_labels = np.random.choice(["Leve", "Moderado", "Grave"], size=len(df), p=[0.5, 0.3, 0.2])
    df["Predicción"] = pred_labels
    
    probabilidades = np.random.dirichlet([1, 1, 1], size=len(df))
    df["Confianza"] = [f"{p.max()*100:.1f}%" for p in probabilidades]
    
    return df

# ======================
# FUNCIONES DE VISUALIZACIÓN
# ======================
def render_kpi_section(df, filtered_df):
    graves = filtered_df[filtered_df["Predicción"] == "Grave"]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reportes", len(filtered_df), delta=f"{len(filtered_df)-len(df)} desde última actualización")
    with col2:
        st.metric("Casos Graves", len(graves), delta_color="inverse")
    with col3:
        productos_count = filtered_df["Producto"].nunique()
        st.metric("Productos Reportados", productos_count)
    with col4:
        avg_response = filtered_df["Días_Inicio_Reacción"].mean() if not filtered_df.empty else 0
        st.metric("Tiempo Reacción Promedio", f"{avg_response:.1f} días")
    
    if not graves.empty:
        st.error(f"🚨 **ALERTA:** Se detectaron {len(graves)} casos graves que requieren atención inmediata!")
        with st.expander("Ver detalles de casos graves"):
            st.dataframe(graves[["ID", "Producto", "Evento_Adverso", "Inicio_Reacción", "Descripción", "Confianza"]])
    
    st.markdown("---")

def render_distribution_tab(filtered_df):
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
            top_products = filtered_df["Producto"].value_counts().nlargest(5).reset_index()
            top_products.columns = ['Producto', 'Reportes']
            
            fig = px.bar(
                top_products, 
                x='Reportes',
                y='Producto',
                orientation='h',
                labels={'Reportes':'Reportes', 'Producto':'Producto'},
                color='Reportes',
                color_continuous_scale="Bluered"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos para mostrar")

def render_trends_tab(filtered_df):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Eventos por Mes")
        if not filtered_df.empty:
            df_mensual = filtered_df.copy()
            df_mensual['Mes'] = df_mensual['Fecha_Reporte'].dt.to_period('M').dt.to_timestamp()
            df_mensual = df_mensual.groupby('Mes').size().reset_index(name='Reportes')
            
            fig = px.line(
                df_mensual, 
                x='Mes',
                y='Reportes',
                labels={'Reportes':'Reportes', 'Mes':'Fecha'},
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

def render_text_tab(filtered_df):
    st.subheader("Análisis de Texto")
    
    if not filtered_df.empty:
        st.write("**Palabras más frecuentes en descripciones**")
        text = " ".join(filtered_df["Descripción"].tolist())
        
        # Create figure safely
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if text.strip():
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                ax.imshow(wordcloud, interpolation='bilinear')
            except:
                ax.text(0.5, 0.5, 'Error generando nube de palabras', 
                         horizontalalignment='center', 
                         verticalalignment='center',
                         fontsize=20)
        else:
            ax.text(0.5, 0.5, 'No hay texto disponible', 
                     horizontalalignment='center', 
                     verticalalignment='center',
                     fontsize=20)
        
        ax.axis("off")
        st.pyplot(fig)
        
        # TF-IDF
        st.write("**Términos más relevantes (TF-IDF)**")
        if text.strip():
            try:
                tfidf = TfidfVectorizer(max_features=10, stop_words=['de', 'la', 'el', 'en', 'y', 'que'])
                X = tfidf.fit_transform(filtered_df["Descripción"])
                tfidf_df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())
                st.bar_chart(tfidf_df.mean().sort_values(ascending=False))
            except:
                st.warning("Error en análisis TF-IDF")
        else:
            st.warning("No hay texto suficiente para análisis TF-IDF")
    else:
        st.warning("No hay datos para mostrar")

def render_map_tab(filtered_df):
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

def render_visualizations(filtered_df):
    st.header("📈 Análisis Visual")
    tab1, tab2, tab3, tab4 = st.tabs(["Distribución", "Tendencias", "Texto", "Mapa"])
    
    with tab1:
        render_distribution_tab(filtered_df)
    with tab2:
        render_trends_tab(filtered_df)
    with tab3:
        render_text_tab(filtered_df)
    with tab4:
        render_map_tab(filtered_df)
    
    st.markdown("---")

def render_sidebar(df):
    with st.sidebar:
        st.header("⚙️ Configuración")
        num_records = st.slider("Número de reportes", 5, 50, 15)
        st.markdown("---")
        
        st.header("🔍 Filtros")
        all_products = st.checkbox("Todos los productos", True, key="all_products")
        severidad_filtro = st.multiselect("Nivel de severidad:", options=["Leve", "Moderado", "Grave"])
        fecha_filtro = st.date_input("Reportes desde:", value=datetime.now() - timedelta(days=180))
        
        productos_filtro = []
        if not all_products:
            productos_filtro = st.multiselect(
                "Productos específicos:", 
                options=df["Producto"].unique() if not df.empty else [],
                default=[]
            )
        
        st.markdown("---")
        st.info("Sophivigil v1.0 | Sistema de monitoreo de eventos adversos oftálmicos")
    
    return num_records, all_products, productos_filtro, severidad_filtro, fecha_filtro

def render_data_section(filtered_df):
    st.header("📋 Datos de Reportes")
    with st.expander("Ver datos completos", expanded=False):
        if not filtered_df.empty:
            # Convertir a string para evitar problemas de formato
            display_df = filtered_df.copy()
            date_cols = ["Inicio_Tratamiento", "Fin_Tratamiento", "Inicio_Reacción", 
                        "Fin_Reacción", "Fecha_Reporte"]
            for col in date_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].dt.strftime('%Y-%m-%d')
            
            st.dataframe(display_df.style.background_gradient(
                subset=["Duración_Reacción", "Duración_Tratamiento"], 
                cmap="YlOrRd"
            ))
        else:
            st.warning("No hay datos disponibles con los filtros actuales")

def render_implementation_plan():
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
# FUNCIÓN PRINCIPAL
# ======================
def main():
    try:
        # Título y separador
        st.title("👁️ Sophivigil – Sistema de Farmacovigilancia Oftálmica")
        st.markdown("---")
        
        # Generar datos iniciales
        df = generate_adr_data(15)
        
        # Barra lateral
        num_records, all_products, productos_filtro, severidad_filtro, fecha_filtro = render_sidebar(df)
        
        # Regenerar datos si cambió el número de registros
        if num_records != len(df):
            df = generate_adr_data(num_records)
        
        # Generar predicciones
        df = generate_predictions(df)
        
        # Aplicar filtros
        filtered_df = apply_filters(df, all_products, productos_filtro, severidad_filtro, fecha_filtro)
        
        # Sección de KPI
        render_kpi_section(df, filtered_df)
        
        # Visualizaciones
        render_visualizations(filtered_df)
        
        # Datos detallados
        render_data_section(filtered_df)
        
        # Plan de implementación
        render_implementation_plan()
        
        # Footer
        st.markdown("---")
        st.caption("© 2023 Sophivigil - Sistema de Farmacovigilancia Oftálmica | Datos simulados con propósitos demostrativos")
    
    except Exception as e:
        st.error(f"**Error crítico en la aplicación**: {str(e)}")
        st.error("Por favor recargue la página o contacte al equipo de soporte")
        st.stop()

# ======================
# EJECUCIÓN PRINCIPAL
# ======================
if __name__ == "__main__":
    main()