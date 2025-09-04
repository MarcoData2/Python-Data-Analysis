import pandas as pd

datos = pd.read_csv("synthetic_ecommerce_data.csv")

# Tipos de datos
print("\nTipos de datos:")
print(datos.dtypes)

# Resumen completo
print("\nResumen del DataFrame:")
print(datos.info())

# Ver primeras filas (para contexto)
print("\nPrimeras filas:")
print(datos.head())