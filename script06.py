import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Dashboard E-commerce Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMetric {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.markdown('<h1 class="main-header">📈 Dashboard Analítico de E-commerce</h1>', unsafe_allow_html=True)
st.markdown("---")

# Carga de datos con manejo mejorado de errores
@st.cache_data
def load_data():
    try:
        datos = pd.read_csv("synthetic_ecommerce_data.csv")
        
        # Transformaciones y limpieza de datos
        date_columns = ['date', 'fecha', 'order_date', 'purchase_date']
        for col in date_columns:
            if col in datos.columns:
                datos[col] = pd.to_datetime(datos[col], errors='coerce')
                datos = datos.dropna(subset=[col])
                datos['month'] = datos[col].dt.month_name()
                datos['day_of_week'] = datos[col].dt.day_name()
                datos['week'] = datos[col].dt.isocalendar().week
                datos['quarter'] = datos[col].dt.quarter
                break
        
        # Identificar columnas de monto/valor
        amount_columns = ['amount', 'price', 'value', 'total', 'revenue']
        for col in amount_columns:
            if col in datos.columns:
                datos[col] = pd.to_numeric(datos[col], errors='coerce')
                datos = datos.dropna(subset=[col])
                # Crear categorías de monto
                if datos[col].max() > 0:
                    bins = [0, 50, 100, 200, 500, np.inf]
                    labels = ['<50', '50-100', '100-200', '200-500', '500+']
                    datos['amount_category'] = pd.cut(datos[col], bins=bins, labels=labels)
                break
        
        return datos
        
    except FileNotFoundError:
        st.error("❌ Archivo 'synthetic_ecommerce_data.csv' no encontrado")
        # Crear datos de ejemplo para demostración
        st.info("💡 Mostrando datos de ejemplo para demostración")
        return create_sample_data()
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
        return create_sample_data()

def create_sample_data():
    """Crear datos de ejemplo para demostración"""
    np.random.seed(42)
    n_rows = 1000
    
    dates = pd.date_range('2024-01-01', '2024-06-30', periods=n_rows)
    categories = ['Electrónicos', 'Ropa', 'Hogar', 'Deportes', 'Juguetes']
    
    data = {
        'date': np.random.choice(dates, n_rows),
        'category': np.random.choice(categories, n_rows, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'amount': np.random.lognormal(4, 1, n_rows),
        'customer_id': [f'CUST_{i:04d}' for i in np.random.randint(1, 201, n_rows)],
        'product_id': [f'PROD_{i:04d}' for i in np.random.randint(1, 51, n_rows)],
        'quantity': np.random.randint(1, 6, n_rows)
    }
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month_name()
    df['day_of_week'] = df['date'].dt.day_name()
    df['week'] = df['date'].dt.isocalendar().week
    
    bins = [0, 50, 100, 200, 500, np.inf]
    labels = ['<50', '50-100', '100-200', '200-500', '500+']
    df['amount_category'] = pd.cut(df['amount'], bins=bins, labels=labels)
    
    return df

# Cargar datos
datos = load_data()

# Sidebar con filtros
st.sidebar.header("🔧 Filtros")
st.sidebar.markdown("Seleccione los filtros para personalizar el análisis:")

# Filtros dinámicos según las columnas disponibles
filtros = {}

# Filtro de categoría
category_cols = [col for col in datos.columns if 'categor' in col.lower() or 'type' in col.lower() or 'category' in col.lower()]
if category_cols:
    cat_col = category_cols[0]
    categorias = st.sidebar.multiselect(
        "Categorías:",
        options=datos[cat_col].unique(),
        default=datos[cat_col].unique()[:3] if len(datos[cat_col].unique()) > 3 else datos[cat_col].unique()
    )
    filtros[cat_col] = categorias

# Filtro de mes
month_cols = [col for col in datos.columns if 'month' in col.lower()]
if month_cols:
    mes_col = month_cols[0]
    meses = st.sidebar.multiselect(
        "Meses:",
        options=datos[mes_col].unique(),
        default=datos[mes_col].unique()[:3] if len(datos[mes_col].unique()) > 3 else datos[mes_col].unique()
    )
    filtros[mes_col] = meses

# Filtro de rango de fechas
date_cols = [col for col in datos.columns if 'date' in col.lower()]
if date_cols:
    fecha_col = date_cols[0]
    min_date = datos[fecha_col].min()
    max_date = datos[fecha_col].max()
    
    fecha_range = st.sidebar.date_input(
        "Rango de fechas:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(fecha_range) == 2:
        filtros[fecha_col] = (pd.to_datetime(fecha_range[0]), pd.to_datetime(fecha_range[1]))

# Filtro de rango de montos
amount_cols = [col for col in datos.columns if any(x in col.lower() for x in ['amount', 'price', 'value', 'total'])]
if amount_cols:
    amount_col = amount_cols[0]
    min_val = float(datos[amount_col].min())
    max_val = float(datos[amount_col].max())
    
    amount_range = st.sidebar.slider(
        f"Rango de {amount_col}:",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val)
    )
    filtros[amount_col] = amount_range

# Aplicar filtros
datos_filtrados = datos.copy()
for col, valores in filtros.items():
    if col in datos_filtrados.columns:
        if isinstance(valores, (list, tuple, np.ndarray)):
            if len(valores) > 0:
                datos_filtrados = datos_filtrados[datos_filtrados[col].isin(valores)]
        elif isinstance(valores, tuple) and len(valores) == 2:
            # Para rangos de fechas o montos
            if pd.api.types.is_datetime64_any_dtype(datos_filtrados[col]):
                datos_filtrados = datos_filtrados[
                    (datos_filtrados[col] >= valores[0]) & 
                    (datos_filtrados[col] <= valores[1])
                ]
            else:
                datos_filtrados = datos_filtrados[
                    (datos_filtrados[col] >= valores[0]) & 
                    (datos_filtrados[col] <= valores[1])
                ]

# Mostrar información del filtrado
st.sidebar.info(f"📊 Datos mostrados: {len(datos_filtrados):,} de {len(datos):,} registros")

# Métricas principales
st.subheader("📊 Métricas Clave")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if amount_cols:
        total_ventas = datos_filtrados[amount_cols[0]].sum()
        st.metric("Ventas Totales", f"${total_ventas:,.2f}", 
                 help="Suma total de todas las transacciones")

with col2:
    customer_cols = [col for col in datos.columns if 'customer' in col.lower() or 'client' in col.lower()]
    if customer_cols:
        clientes_unicos = datos_filtrados[customer_cols[0]].nunique()
        st.metric("Clientes Únicos", f"{clientes_unicos:,}",
                 help="Número total de clientes distintos")

with col3:
    product_cols = [col for col in datos.columns if 'product' in col.lower() or 'item' in col.lower()]
    if product_cols:
        productos_unicos = datos_filtrados[product_cols[0]].nunique()
        st.metric("Productos Únicos", f"{productos_unicos:,}",
                 help="Número total de productos distintos")

with col4:
    if amount_cols and customer_cols:
        ticket_promedio = datos_filtrados[amount_cols[0]].sum() / datos_filtrados[customer_cols[0]].nunique()
        st.metric("Ticket Promedio", f"${ticket_promedio:,.2f}",
                 help="Gasto promedio por cliente")

st.markdown("---")

# Visualizaciones
st.subheader("📈 Análisis Visual")

# Layout de gráficos
col1, col2 = st.columns(2)

with col1:
    # Gráfico de ventas por categoría
    if category_cols and amount_cols:
        ventas_por_categoria = datos_filtrados.groupby(category_cols[0])[amount_cols[0]].sum().reset_index()
        ventas_por_categoria = ventas_por_categoria.sort_values(amount_cols[0], ascending=False)
        
        fig = px.bar(ventas_por_categoria, x=category_cols[0], y=amount_cols[0], 
                     title='Ventas por Categoría', color=category_cols[0],
                     labels={amount_cols[0]: 'Ventas Totales', category_cols[0]: 'Categoría'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Serie temporal de ventas
    if date_cols and amount_cols:
        ventas_por_fecha = datos_filtrados.groupby(date_cols[0])[amount_cols[0]].sum().reset_index()
        fig = px.line(ventas_por_fecha, x=date_cols[0], y=amount_cols[0], 
                      title='Evolución de Ventas en el Tiempo',
                      labels={amount_cols[0]: 'Ventas', date_cols[0]: 'Fecha'})
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # Gráfico de distribución de montos
    if amount_cols:
        fig = px.histogram(datos_filtrados, x=amount_cols[0], 
                           title='Distribución de Montos de Transacción', 
                           nbins=20,
                           labels={amount_cols[0]: 'Monto'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de proporción por categoría
    if category_cols:
        conteo_categorias = datos_filtrados[category_cols[0]].value_counts().reset_index()
        fig = px.pie(conteo_categorias, values='count', names=category_cols[0], 
                     title='Proporción de Transacciones por Categoría')
        st.plotly_chart(fig, use_container_width=True)

# Heatmap de días de la semana vs meses
if month_cols and 'day_of_week' in datos_filtrados.columns and amount_cols:
    st.subheader("🔥 Análisis de Tendencia: Ventas por Día y Mes")
    
    # Crear heatmap data
    heatmap_data = datos_filtrados.groupby([month_cols[0], 'day_of_week'])[amount_cols[0]].sum().unstack()
    
    # Ordenar días de la semana y meses
    dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    meses_orden = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    # Reindexar para ordenar correctamente
    heatmap_data = heatmap_data.reindex(
        index=[m for m in meses_orden if m in heatmap_data.index], 
        columns=[d for d in dias_orden if d in heatmap_data.columns]
    )
    
    fig = px.imshow(heatmap_data, aspect="auto", 
                    title="Ventas por Día de la Semana y Mes",
                    labels=dict(x="Día de la Semana", y="Mes", color="Ventas"))
    st.plotly_chart(fig, use_container_width=True)

# Análisis de productos más vendidos
if product_cols:
    st.subheader("🏆 Productos Más Vendidos")
    top_productos = datos_filtrados[product_cols[0]].value_counts().head(10).reset_index()
    fig = px.bar(top_productos, x='count', y=product_cols[0], orientation='h',
                 title='Top 10 Productos por Cantidad de Ventas',
                 labels={'count': 'Cantidad de Ventas', product_cols[0]: 'Producto'})
    st.plotly_chart(fig, use_container_width=True)

# Información detallada del dataset
with st.expander("📋 Ver Detalles Técnicos del Dataset"):
    st.subheader("Resumen del Dataset")
    st.write(f"**Dimensiones:** {datos.shape[0]:,} filas × {datos.shape[1]} columnas")
    st.write(f"**Período de datos:** {datos[date_cols[0]].min().strftime('%Y-%m-%d') if date_cols else 'N/A'} to {datos[date_cols[0]].max().strftime('%Y-%m-%d') if date_cols else 'N/A'}")
    
    st.subheader("Tipos de Datos y Valores Nulos")
    
    # Crear resumen evitando tipos de datos problemáticos
    resumen_tipos = pd.DataFrame({
        'Columna': datos.columns,
        'Tipo de Dato': [str(dtype) for dtype in datos.dtypes],
        'Valores Nulos': datos.isnull().sum().values,
        '% Nulos': (datos.isnull().sum() / len(datos) * 100).round(2).values
    })
    
    # Mostrar como tabla de Streamlit en lugar de dataframe
    st.table(resumen_tipos)
    
    st.subheader("Estadísticas Descriptivas")
    if amount_cols:
        st.write(datos[amount_cols[0]].describe())
    
    st.subheader("Muestra de Datos")
    st.dataframe(datos.head(), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("📊 *Dashboard creado con Streamlit y Plotly* • 🕐 *Actualizado: Septiembre 2024*")

# Información de uso
st.sidebar.markdown("---")
st.sidebar.info("""
**💡 Tips de uso:**
- Use los filtros para segmentar los datos
- Haga hover sobre los gráficos para ver detalles
- Expanda la sección de detalles técnicos para metadatos
""")