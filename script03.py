import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuración de estilo para gráficos
plt.style.use('default')
sns.set_palette("husl")

# 1. CARGAR Y EXPLORAR EL DATASET
print("="*60)
print("CARGANDO Y EXPLORANDO EL DATASET")
print("="*60)

# Cargar el dataset
df = pd.read_csv('synthetic_ecommerce_data.csv')

# Mostrar información inicial
print(f"Dimensiones iniciales: {df.shape}")
print(f"\nPrimeras 5 filas:")
print(df.head())
print(f"\nTipos de datos iniciales:")
print(df.dtypes)
print(f"\nValores nulos iniciales:")
print(df.isnull().sum())

# 2. LIMPIEZA DE DATOS
print("\n" + "="*60)
print("PROCESO DE LIMPIEZA")
print("="*60)

# Convertir OrderDate a datetime
df['OrderDate'] = pd.to_datetime(df['OrderDate'])

# Normalizar Gender (primera letra mayúscula)
df['Gender'] = df['Gender'].str.capitalize()

# Limpiar espacios en Location y ProductCategory
df['Location'] = df['Location'].str.strip()
df['ProductCategory'] = df['ProductCategory'].str.strip()

# Crear nuevas características
df['AgeGroup'] = pd.cut(df['Age'], 
                       bins=[0, 25, 35, 45, 55, 65, 100],
                       labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66+'])

df['OrderMonth'] = df['OrderDate'].dt.month
df['OrderYear'] = df['OrderDate'].dt.year
df['OrderDay'] = df['OrderDate'].dt.day
df['OrderWeekday'] = df['OrderDate'].dt.day_name()

# Validar datos numéricos
df = df[df['Price'] >= 0]
df = df[df['Quantity'] > 0]
df = df[(df['ReviewScore'] >= 1) & (df['ReviewScore'] <= 5)]

# Verificar consistencia de TotalAmount
df['CalculatedTotal'] = df['Price'] * df['Quantity']
inconsistencies = df[abs(df['TotalAmount'] - df['CalculatedTotal']) > 0.01]
if len(inconsistencies) > 0:
    print(f"Corrigiendo {len(inconsistencies)} inconsistencias en TotalAmount")
    df['TotalAmount'] = df['Price'] * df['Quantity']

df = df.drop('CalculatedTotal', axis=1)

# Normalizar métodos de pago
payment_mapping = {
    'Digital Wallet': 'Digital',
    'Credit Card': 'Credit_Card',
    'Debit Card': 'Debit_Card',
    'PayPal': 'PayPal'
}
df['PaymentMethod'] = df['PaymentMethod'].map(payment_mapping)

# Eliminar duplicados y resetear índice
df = df.drop_duplicates().reset_index(drop=True)

# 3. INFORMACIÓN FINAL DESPUÉS DE LIMPIEZA
print("\n" + "="*60)
print("DATASET DESPUÉS DE LA LIMPIEZA")
print("="*60)

print(f"Dimensiones finales: {df.shape}")
print(f"\nTipos de datos actualizados:")
print(df.dtypes)
print(f"\nValores nulos:")
print(df.isnull().sum())

print(f"\nDistribución de Género:")
print(df['Gender'].value_counts())
print(f"\nDistribución de Categorías de Producto:")
print(df['ProductCategory'].value_counts())
print(f"\nRango de fechas: {df['OrderDate'].min()} to {df['OrderDate'].max()}")

# 4. ANÁLISIS EXPLORATORIO
print("\n" + "="*60)
print("ANÁLISIS EXPLORATORIO")
print("="*60)

# Crear gráficos
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('ANÁLISIS EXPLORATORIO - DATOS DE E-COMMERCE', fontsize=16, fontweight='bold')

# Gráfico 1: Distribución de Edad
axes[0, 0].hist(df['Age'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribución de Edad de Clientes', fontweight='bold')
axes[0, 0].set_xlabel('Edad')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].grid(True, alpha=0.3)

# Gráfico 2: Ventas por Categoría
category_sales = df.groupby('ProductCategory')['TotalAmount'].sum().sort_values(ascending=False)
axes[0, 1].bar(category_sales.index, category_sales.values, color='lightgreen', alpha=0.7)
axes[0, 1].set_title('Ventas Totales por Categoría', fontweight='bold')
axes[0, 1].set_xlabel('Categoría')
axes[0, 1].set_ylabel('Ventas Totales ($)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# Gráfico 3: Métodos de Pago
payment_counts = df['PaymentMethod'].value_counts()
axes[1, 0].pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%', 
               colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
axes[1, 0].set_title('Distribución de Métodos de Pago', fontweight='bold')

# Gráfico 4: Scores de Reviews
review_counts = df['ReviewScore'].value_counts().sort_index()
axes[1, 1].bar(review_counts.index.astype(str), review_counts.values, color='orange', alpha=0.7)
axes[1, 1].set_title('Distribución de Review Scores', fontweight='bold')
axes[1, 1].set_xlabel('Score')
axes[1, 1].set_ylabel('Cantidad')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. ANÁLISIS TEMPORAL
plt.figure(figsize=(12, 6))

# Ventas mensuales
monthly_sales = df.groupby(['OrderYear', 'OrderMonth'])['TotalAmount'].sum().reset_index()
monthly_sales['Date'] = pd.to_datetime(monthly_sales['OrderYear'].astype(str) + '-' + 
                                      monthly_sales['OrderMonth'].astype(str).str.zfill(2))

plt.plot(monthly_sales['Date'], monthly_sales['TotalAmount'], 
         marker='o', linewidth=2, markersize=4, color='purple')
plt.title('Evolución de Ventas Mensuales', fontsize=14, fontweight='bold')
plt.xlabel('Mes')
plt.ylabel('Ventas Totales ($)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. MATRIZ DE CORRELACIÓN
numeric_cols = ['Age', 'Price', 'Quantity', 'ReviewScore', 'TotalAmount']
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Matriz de Correlación entre Variables Numéricas', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 7. ESTADÍSTICAS FINALES
print("\n" + "="*60)
print("ESTADÍSTICAS FINALES")
print("="*60)

print(f"Total de ventas: ${df['TotalAmount'].sum():,.2f}")
print(f"Venta promedio por transacción: ${df['TotalAmount'].mean():,.2f}")
print(f"Transacciones totales: {len(df):,}")
print(f"Clientes únicos: {df['CustomerID'].nunique():,}")
print(f"Productos únicos: {df['ProductID'].nunique():,}")

print(f"\nTicket promedio por método de pago:")
print(df.groupby('PaymentMethod')['TotalAmount'].mean().round(2))

print(f"\nReview score promedio: {df['ReviewScore'].mean():.2f}")

# 8. GUARDAR DATASET LIMPIO
df.to_csv('ecommerce_data_cleaned.csv', index=False)
print(f"\nDataset limpio guardado como 'ecommerce_data_cleaned.csv'")

# 9. MUESTRA FINAL
print("\n" + "="*60)
print("MUESTRA FINAL DEL DATASET LIMPIO")
print("="*60)
print(df.sample(3))

print("\n¡Proceso completado exitosamente! ✅")