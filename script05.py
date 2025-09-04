import pandas as pd
import numpy as np

# Configuración para mejor visualización
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)

# Cargar datos con manejo de errores
try:
    datos = pd.read_csv("synthetic_ecommerce_data.csv")
    print("✅ Datos cargados exitosamente")
except FileNotFoundError:
    print("❌ Error: Archivo no encontrado")
    exit()
except Exception as e:
    print(f"❌ Error al cargar el archivo: {e}")
    exit()

# 1. INFORMACIÓN BÁSICA DEL DATASET
print("="*60)
print("ANÁLISIS EXPLORATORIO DE DATOS - ECOMMERCE")
print("="*60)

print(f"\n📊 DIMENSIONES DEL DATASET:")
print(f"• Filas: {datos.shape[0]:,}")
print(f"• Columnas: {datos.shape[1]}")
print(f"• Total de valores: {datos.size:,}")

# 2. TIPOS DE DATOS Y VALORES NULOS
print(f"\n🔍 TIPOS DE DATOS Y VALORES NULOS:")
print("-"*40)

# Crear resumen de tipos de datos y nulos
resumen_tipos = pd.DataFrame({
    'Tipo_Dato': datos.dtypes,
    'Valores_Nulos': datos.isnull().sum(),
    'Porcentaje_Nulos': (datos.isnull().sum() / len(datos) * 100).round(2)
})
print(resumen_tipos)

# 3. ESTADÍSTICAS DESCRIPTIVAS
print(f"\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
print("-"*40)

# Separar columnas numéricas y categóricas
columnas_numericas = datos.select_dtypes(include=[np.number]).columns
columnas_categoricas = datos.select_dtypes(include=['object']).columns

if len(columnas_numericas) > 0:
    print("Variables numéricas:")
    print(datos[columnas_numericas].describe().round(2))
    
if len(columnas_categoricas) > 0:
    print("\nVariables categóricas:")
    for col in columnas_categoricas:
        print(f"\n{col}:")
        print(f"• Valores únicos: {datos[col].nunique()}")
        print(f"• Top 5 valores:")
        print(datos[col].value_counts().head())

# 4. MUESTRA DE DATOS
print(f"\n👀 MUESTRA DE DATOS:")
print("-"*40)

print("Primeras 5 filas:")
print(datos.head())
print(f"\nÚltimas 3 filas:")
print(datos.tail(3))
print(f"\nMuestra aleatoria de 5 filas:")
print(datos.sample(5, random_state=42))

# 5. INFORMACIÓN ADICIONAL
print(f"\nℹ️ INFORMACIÓN COMPLEMENTARIA:")
print("-"*40)

print(f"• Memoria utilizada: {datos.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"• Columnas duplicadas: {datos.duplicated().sum()}")
if datos.duplicated().sum() > 0:
    print("  ⚠️  Se encontraron filas duplicadas")

# 6. ANÁLISIS DE FECHAS (si existen)
fechas_cols = datos.select_dtypes(include=['datetime64']).columns
if len(fechas_cols) > 0:
    print(f"\n📅 VARIABLES DE FECHA:")
    for col in fechas_cols:
        print(f"{col}: {datos[col].min()} to {datos[col].max()}")

# Guardar resumen en archivo
try:
    with open('resumen_analisis.txt', 'w') as f:
        f.write("RESUMEN DE ANÁLISIS EXPLORATORIO\n")
        f.write("="*50 + "\n")
        f.write(f"Filas: {datos.shape[0]}, Columnas: {datos.shape[1]}\n\n")
        f.write(str(resumen_tipos))
    print(f"\n💾 Resumen guardado en 'resumen_analisis.txt'")
except Exception as e:
    print(f"⚠️  No se pudo guardar el resumen: {e}")

print("\n" + "="*60)
print("ANÁLISIS COMPLETADO ✅")
print("="*60)