# Análisis y Predicción de la Producción de Electricidad con Energías Renovables

Este proyecto tiene como objetivo analizar y predecir la producción de electricidad, con especial énfasis en las energías renovables, utilizando el conjunto de datos "Month Electricity Statistics" de la Agencia Internacional de Energía (IEA).

## Objetivo General

El código fuente de la aplicación en Streamlit realiza las siguientes tareas:

- **Carga y preprocesamiento de los datos:**  
  Se conecta a una base de datos SQLite que contiene los datos de la IEA, se leen en un DataFrame de pandas y se preparan para el análisis y la visualización.

- **Visualización de series de tiempo:**  
  Se generan gráficos interactivos (líneas, barras, etc.) que permiten explorar las tendencias en la producción de electricidad en función de variables como país, balance y producto energético.

- **Análisis descriptivo:**  
  Se calculan estadísticas descriptivas (media, desviación estándar, cuartiles, etc.) y se visualiza la distribución de los datos para comprender las características del conjunto de datos.

- **Análisis exploratorio:**  
  Se focaliza el estudio en las energías renovables para responder a la pregunta: ¿Cuál energía renovable tiene el mayor margen de crecimiento?  
  Se calcula la tasa de crecimiento anual compuesto (CAGR) para cada país y se visualiza mediante gráficos de barras horizontales, identificando potenciales oportunidades de crecimiento.

- **Predicción de la producción de electricidad:**  
  Se utiliza el modelo ARIMA (con normalización Min-Max) para predecir la producción de electricidad en el futuro.  
  El usuario puede seleccionar los parámetros `p`, `d` y `q` para ajustar el modelo a sus necesidades.

- **Evaluación del modelo y visualización de resultados:**  
  Se evalúa la precisión del modelo mediante métricas como MAE, RMSE y R².  
  Los resultados se visualizan en gráficos comparativos que muestran datos reales y predicciones.  
  Además, se ofrece la opción de descargar los resultados en formato CSV.

## Detalles Específicos

### 1. Carga y Preprocesamiento de Datos

- **Conexión a la Base de Datos:**  
  Se utiliza la librería `sqlite3` para conectar a la base de datos SQLite y `pandas` para leer los datos de la vista `energy_view_final`.

- **Conversión de Fechas y Limpieza:**  
  La columna `Time` se convierte a formato datetime. Se eliminan los valores nulos y se aplican filtros por país, balance y producto energético.  
  Existe la opción de eliminar valores atípicos utilizando el método del IQR.

### 2. Visualización de Series de Tiempo

- **Herramientas de Visualización:**  
  Se usan `plotly.express` y `plotly.graph_objects` para generar gráficos interactivos que permiten:
  - Comparar la producción de electricidad entre distintos países, balances y productos.
  - Visualizar tendencias y patrones a lo largo del tiempo.

### 3. Análisis Descriptivo

- **Estadísticas Básicas:**  
  Se utiliza la función `describe()` de pandas para obtener un resumen estadístico de la columna `Value`.  
- **Distribución y Calidad de Datos:**  
  Se explora la distribución, valores nulos y duplicados para comprender mejor el conjunto de datos.

### 4. Análisis Exploratorio

- **Enfoque en Energías Renovables:**  
  Se calcula la tasa de crecimiento anual compuesto (CAGR) para cada país y se analiza cuál energía renovable tiene mayor potencial de crecimiento.
- **Casos Destacados:**  
  Se realiza un análisis específico de la energía solar en Colombia, destacando el notable crecimiento registrado en el mes de Octubre de 2024.
- **Visualizaciones:**  
  Gráficos de barras horizontales permiten comparar visualmente el CAGR entre los países y energías.

### 5. Predicción de la Producción de Electricidad

- **Modelo ARIMA:**  
  Se implementa un modelo ARIMA (usando la librería `statsmodels`) para predecir la producción de electricidad.  
- **Normalización de Datos:**  
  Se utiliza `MinMaxScaler` de `sklearn.preprocessing` para escalar la columna `Value` y mejorar el desempeño del modelo.
- **Ajuste y Pronóstico:**  
  El modelo se entrena con datos históricos y se utiliza para generar predicciones in-sample y out-of-sample, según los parámetros seleccionados por el usuario.

### 6. Evaluación del Modelo y Visualización de Resultados

- **Métricas de Evaluación:**  
  Se calculan el MAE, RMSE y R² para evaluar la precisión del modelo.
- **Comparación de Resultados:**  
  Los resultados de la predicción se visualizan junto a los datos reales en gráficos interactivos.
- **Descarga de Resultados:**  
  Se proporciona la opción de descargar los resultados en formato CSV.
- **Interpretación:**  
  Se incluyen descripciones y analogías para facilitar la interpretación de los elementos clave del modelo ARIMA y las métricas de evaluación.

## En Resumen

El código fuente de esta aplicación permite:

- Explorar y analizar la producción de electricidad con énfasis en las energías renovables.
- Identificar las energías con mayor potencial de crecimiento.
- Predecir la producción futura mediante modelos ARIMA.
- Evaluar la precisión del modelo a través de métricas y visualizaciones interactivas.

El objetivo final es proporcionar información valiosa para la toma de decisiones en el sector energético, ayudando a planificar inversiones, optimizar la generación y mejorar la respuesta a la demanda futura.

## Referencias

- **Agencia Internacional de Energía [IEA](https://www.iea.org/)**
- **Documentación de [Streamlit](https://docs.streamlit.io/)**
- **Documentación de [Plotly](https://plotly.com/python/)**
- **Documentación de [statsmodels](https://www.statsmodels.org/)**
- **Documentación de [scikit-learn](https://scikit-learn.org/stable/)**

---

