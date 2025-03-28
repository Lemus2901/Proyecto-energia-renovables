import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler



# Configuración básica de la página (única para toda la aplicación)
st.set_page_config(
    page_title="Electricidad - Proyecto Bootcamp ARIMA",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para definir los colores: azul por defecto, verde para el valor máximo y rojo para el mínimo.
def get_colors(counts):
    # Obtenemos los índices donde se encuentra el máximo y mínimo
    max_index = counts.idxmax()
    min_index = counts.idxmin()
    # Generamos la lista de colores acorde a cada valor
    colors = ['green' if idx == max_index else 'red' if idx == min_index else 'blue' for idx in counts.index]
    return colors

def vista_prediccion():
    st.header('Vista de Predicción de Value con ARIMA y normalización Min-Max - Dataset Month Electricity Statistics (IEA)')
    #st.warning('Se debe cargar un archivo con la BD y las columnas requeridas.')

    # Cargar datos desde Excel
    uploaded_file = st.file_uploader("Elige un archivo de Base de Datos (Sqlite3)", type=["db", "sqlite3"])
    if uploaded_file:
        try:
            
            temp_db_path = "energyiea.db"
    
            # Guardamos el archivo subido en disco
            with open(temp_db_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success("Archivo subido correctamente.")
            
            # Conexión a la base de datos SQLite
            conn_sqlite = sqlite3.connect(temp_db_path)  # Reemplaza con el nombre de tu archivo SQLite

            # Leer la vista 'energy_view' en un DataFrame de pandas
            df = pd.read_sql_query("SELECT * FROM energy_view_final", conn_sqlite)

            # Convertir la columna 'Time' a datetime
            df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d')
            
            df.dropna(subset=['Value'], inplace=True)

            # Filtros de selección
            country = st.selectbox("Selecciona un país:", df['Country'].unique(), index=40)
            balance = st.selectbox("Selecciona un Balance:", df['Balance'].unique(), index=0)
            product = st.selectbox("Selecciona un Producto:", df['Product'].unique(), index=8)
            value_col = st.selectbox("Selecciona la columna de valor:", ['Value'])

            # Filtrar DataFrame
            df_filtered = df[(df['Country'] == country) &
                             (df['Balance'] == balance) &
                             (df['Product'] == product)]

            # Controles para el periodo de tiempo
            min_date = df_filtered['Time'].min()
            max_date = df_filtered['Time'].max()

            start_year = st.selectbox("Año de inicio:", range(min_date.year, max_date.year + 1), index=0)
            start_month = st.selectbox("Mes de inicio:", range(1, 13), index=min_date.month - 1)
            start_date = pd.to_datetime(f'{start_year}-{start_month}-01')

            end_year = st.selectbox("Año de fin:", range(start_date.year, max_date.year + 1), 
                                    index=len(range(start_date.year, max_date.year + 1))-1)
            end_month = st.selectbox("Mes de fin:", range(1, 13), index=max_date.month - 1)
            end_date = pd.to_datetime(f'{end_year}-{end_month}-01')

            # Filtrar por periodo de tiempo
            df_filtered = df_filtered[(df_filtered['Time'] >= start_date) & (df_filtered['Time'] <= end_date)]

            # Opción para eliminar valores atípicos (IQR)
            if st.checkbox("Eliminar valores atípicos (IQR)"):
                Q1 = df_filtered[value_col].quantile(0.25)
                Q3 = df_filtered[value_col].quantile(0.75)
                IQR = Q3 - Q1
                df_filtered = df_filtered[~((df_filtered[value_col] < (Q1 - 1.5 * IQR)) | 
                                            (df_filtered[value_col] > (Q3 + 1.5 * IQR)))]

            # Mostrar DataFrame filtrado
            st.dataframe(df_filtered)

            # Parámetros para el modelo ARIMA y el pronóstico
            periodos_futuros = st.slider('Periodos a predecir (meses)', min_value=1, max_value=24, step=1, value=6)
            p = st.number_input("ARIMA: p (orden autorregresivo)", min_value=0, max_value=10, value=0, step=1)
            d = st.number_input("ARIMA: d (orden de diferenciación)", min_value=0, max_value=2, value=2, step=1)
            q = st.number_input("ARIMA: q (orden de media móvil)", min_value=0, max_value=10, value=1, step=1)

            if st.button('Ejecutar predicción', type='primary'):
                try:
                    # Ordenar los datos por fecha y reiniciar el índice
                    df_filtered = df_filtered.sort_values('Time').reset_index(drop=True)

                    # Normalización Min-Max de la columna Value
                    scaler = MinMaxScaler()
                    df_filtered['Value_norm'] = scaler.fit_transform(df_filtered[[value_col]])

                    # Preparar la serie temporal y las fechas
                    series = df_filtered['Value_norm']
                    dates = df_filtered['Time']

                    # Ajustar el modelo ARIMA en los datos normalizados
                    model = ARIMA(series, order=(p, d, q))
                    model_fit = model.fit(cov_type="robust")

                    # Realizar predicción in-sample (desde el punto donde se pueden obtener predicciones)
                    start_pred = d  # Los primeros 'd' valores pueden no tener predicción
                    pred_in_sample = model_fit.predict(start=start_pred, end=len(series)-1)

                    # Invertir la normalización para obtener los valores en escala original
                    actual_in_sample = scaler.inverse_transform(series[start_pred:].values.reshape(-1, 1)).flatten()
                    pred_in_sample_original = scaler.inverse_transform(pred_in_sample.values.reshape(-1, 1)).flatten()

                    # Calcular métricas de error con la predicción in-sample
                    mae = mean_absolute_error(actual_in_sample, pred_in_sample_original)
                    rmse = np.sqrt(mean_squared_error(actual_in_sample, pred_in_sample_original))
                    r2 = r2_score(actual_in_sample, pred_in_sample_original)

                    # Pronóstico de periodos futuros
                    forecast_norm = model_fit.forecast(steps=periodos_futuros)
                    forecast_original = scaler.inverse_transform(forecast_norm.values.reshape(-1, 1)).flatten()

                    # Generar fechas para el pronóstico futuro (datos mensuales)
                    last_date = dates.iloc[-1]
                    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                                 periods=periodos_futuros, freq='MS')

                    # Preparar DataFrames para resultados:
                    # Datos reales
                    df_actual = pd.DataFrame({
                        'ds': dates,
                        'y': df_filtered[value_col],
                        'Tipo': 'Real'
                    })

                    # Pronóstico futuro
                    df_forecast = pd.DataFrame({
                        'ds': future_dates,
                        'y': forecast_original,
                        'Tipo': 'Predicción'
                    })

                    # Predicción in-sample (para evaluar el modelo)
                    df_insample = pd.DataFrame({
                        'ds': dates[start_pred:],
                        'y': pred_in_sample_original,
                        'Tipo': 'Predicción In-Sample'
                    })

                    # Combinar datos reales y pronóstico futuro para la visualización
                    df_resultado = pd.concat([df_actual, df_forecast])

                    # Mostrar resultados en pestañas
                    tab3, tab2, tab1 = st.tabs(['Resumen ARIMA', 'Gráficos', 'Resultado'])
                    with tab1:
                        col1, col2 = st.columns([30, 70])
                        with col1:
                            st.subheader("Tabla de resultados")
                            st.dataframe(df_resultado)
                            csv = df_resultado.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Descargar resultado como CSV",
                                data=csv,
                                file_name=f'prediccion_arima_{uploaded_file.name}',
                                mime='text/csv'
                            )

                            # Mostrar métricas de error
                            metrics_df = pd.DataFrame({
                                'Métrica': ['MAE', 'RMSE', 'R2'],
                                'Valor': [mae, rmse, r2],
                                'Descripción': [
                                    f"Error promedio de {mae:.2f} GWh.",
                                    f"Error cuadrático medio de {rmse:.2f} GWh.",
                                    f"R2 de {r2:.2f} (cercano a 1 indica buen ajuste)."
                                ]
                            })
                            st.subheader("Métricas de Precisión (In-Sample)")
                            st.dataframe(metrics_df)

                        with col2:
                            st.write("Gráfica de la serie real y el pronóstico de ARIMA:")
                            fig = px.line(df_resultado, x='ds', y='y', color='Tipo')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("""
                            > **Imagina que tu modelo ARIMA es como un mapa básico que muestra la ruta de crecimiento de la producción solar en Colombia.**  
                            > Aunque no sea el mapa más detallado (pues no incluye todas las curvas o cambios de clima), sí te permite ver hacia dónde se dirige el camino principal y con qué pendiente. Este panorama inicial puede ser suficiente para animar a los responsables de políticas públicas a invertir en infraestructura y programas de fomento, al ver una clara tendencia ascendente en la generación de GWh solares.
                            >
                            > **Ahora bien, no olvidemos que este mapa se ha dibujado con la información disponible en un solo momento (in-sample).**  
                            > Si queremos estar realmente seguros de que la ruta es la correcta y de que no nos esperan desvíos bruscos más adelante, necesitamos datos fuera de la muestra (out-of-sample). Es como pedirle a otras personas que recorran parte del camino y nos confirmen si el trazo que hicimos coincide con la realidad. Al hacerlo, tendremos una visión más sólida de la precisión de nuestro modelo y podremos ajustar mejor los parámetros (como p, d y q) o incluso incluir nuevos factores (por ejemplo, costos de paneles o políticas energéticas) que enriquezcan nuestras predicciones.
                            >
                            > **Por eso, te invitamos a compartir y recopilar datos recientes o adicionales que no hayan sido usados en la fase de entrenamiento.**  
                            > Al incorporar esa información, lograremos un modelo que no solo muestre la tendencia, sino que también sea más confiable y capaz de guiar decisiones de inversión, regulación y desarrollo tecnológico. Así, entre todos, contribuiremos a que la energía solar en Colombia continúe su trayectoria de crecimiento con bases cada vez más sólidas.
                            """)

                    with tab2:
                        st.subheader("Comparación: Real vs Predicción In-Sample")
                        df_compare = df_actual.merge(df_insample, on='ds', how='inner', suffixes=('_Real', '_Pred'))
                        fig2 = px.line(df_compare, x='ds', y=['y_Real', 'y_Pred'], 
                                       labels={'value': 'Valor', 'variable': 'Tipo'})
                        st.plotly_chart(fig2, use_container_width=True)
                        
                    with tab3:
                        st.header("Resumen de Elementos Clave para el Modelo ARIMA Actual")
                        resumen = pd.DataFrame({
                            "Elemento": [
                                "Estacionariedad (d)",
                                "Orden Autoregresivo (p)",
                                "Orden de Media Móvil (q)"
                            ],
                            "Descripción y Ejemplo": [
                                "La serie se estabiliza mediante diferenciación. En este modelo se utiliza, por ejemplo, d=2 para intentar estabilizar la media y la varianza sin recurrir a pruebas de estacionariedad (ADF).",
                                "Determina la influencia de valores pasados. Si la serie muestra poca dependencia en sus valores previos, p=0 puede simplificar el modelo sin pérdida de información.",
                                "Modela el efecto de errores pasados. Por ejemplo, si el análisis sugiere que los errores tienen cierta persistencia, usar q=1 puede capturar adecuadamente esta dinámica."
                            ]
                        })
                        st.table(resumen)
                        
                        st.header("Valor Aportado por la Predicción")
                        pred_valor = pd.DataFrame({
                            "Aspecto": [
                                "Exactitud (MAE y RMSE)",
                                "Capacidad Explicativa (R²)"
                            ],
                            "Analogía y Ejemplo": [
                                "Piensa en MAE y RMSE como una brújula: te indican, en promedio, cuán lejos se desvían las predicciones del rumbo correcto. RMSE penaliza más los errores grandes, similar a notar un gran desvío en un viaje.",
                                "R² es como armar un rompecabezas: un valor cercano a 1 indica que casi todas las piezas del panorama se capturan, generando confianza en que el modelo refleja la realidad para la toma de decisiones."
                            ]
                        })
                        st.table(pred_valor)

                except Exception as e:
                    st.error(f"Error durante la predicción: {e}")
        except Exception as e:
            st.error(f"Error al cargar el archivo .db o procesar los datos: {e}")
    else:
        st.write("Por favor, sube un archivo .db para comenzar.")

def vista_graficos():
    st.header("Vista de Series de Tiempo - Dataset Month Electricity Statistics (IEA)")
    #st.warning('Se debe cargar un archivo Excel con una hoja llamada "TableData" y las columnas requeridas.')

    #uploaded_file = st.file_uploader("Elige un archivo Excel", type=["xlsx"], key="graficos")
    # Por favor, sube un archivo Excel para visualizar los gráficos.
    # Cargar datos desde Excel
    uploaded_file = st.file_uploader("Elige un archivo de Base de Datos (Sqlite3)", type=["db", "sqlite3"])
    if uploaded_file:
        try:
            
            temp_db_path = "energyiea.db"
    
            # Guardamos el archivo subido en disco
            with open(temp_db_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success("Archivo subido correctamente.")
            
            # Conexión a la base de datos SQLite
            conn_sqlite = sqlite3.connect(temp_db_path)  # Reemplaza con el nombre de tu archivo SQLite

            # Leer la vista 'energy_view' en un DataFrame de pandas
            df = pd.read_sql_query("SELECT * FROM energy_view_final", conn_sqlite)
            
            df.dropna(subset=['Value'], inplace=True)
    #if uploaded_file:
        #try:
            #df = pd.read_excel(uploaded_file, sheet_name="TableData")
            
            # Convertir la columna 'Time' a datetime si existe
            if 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'])
            else:
                st.error("El archivo debe contener la columna 'Time'.")
                return

            # Verificar que 'Value' esté presente
            if 'Value' not in df.columns:
                st.error("El archivo debe contener la columna 'Value'.")
                return

            # Filtros múltiples (sin selección por defecto)
            countries = st.multiselect(
                "Selecciona países",
                options=df['Country'].unique(),
                default=[]
            )
            balances = st.multiselect(
                "Selecciona balances",
                options=df['Balance'].unique(),
                default=[]
            )
            products = st.multiselect(
                "Selecciona productos",
                options=df['Product'].unique(),
                default=[]
            )

            # Filtrar el DataFrame según selecciones
            df_filtered = df[
                df['Country'].isin(countries) &
                df['Balance'].isin(balances) &
                df['Product'].isin(products)
            ]

            # Seleccionar el tipo de gráfico
            grafico_tipo = st.selectbox("Selecciona el tipo de gráfico", ["Línea", "Barras"])

            if grafico_tipo == "Línea":
                # Gráfico de línea: una serie distinta por cada combinación de País, Balance y Producto
                fig = px.line(
                    df_filtered,
                    x="Time",
                    y="Value",
                    color="Country",       # Diferencia por País
                    line_dash="Balance",   # Diferencia por Balance (estilo de línea)
                    symbol="Product",      # Diferencia por Producto (símbolo)
                    title="Valor a lo largo del tiempo"
                )
                st.plotly_chart(fig, use_container_width=True)

            elif grafico_tipo == "Barras":
                # Agrupar por País, Balance y Producto para calcular promedio
                df_grouped = df_filtered.groupby(["Country", "Balance", "Product"])['Value'].mean().reset_index()

                # Gráfico de barras con facetas por Balance y color por País
                fig = px.bar(
                    df_grouped,
                    x="Product",
                    y="Value",
                    color="Country",
                    facet_col="Balance",
                    title="Valor promedio por País, Balance y Producto"
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
    else:
        st.write("Por favor, sube un archivo .db para visualizar los gráficos.")
        
def vista_analisis_descriptivo():
    """
    Crea un mapa dinámico con filtros interactivos.
    """

    st.header("Análisis Descriptivo - Dataset Month Electricity Statistics (IEA)")

    # Cargar datos desde SQLite
    uploaded_file = st.file_uploader("Elige un archivo de Base de Datos (Sqlite3)", type=["db", "sqlite3"])
    if uploaded_file:
        try:
            temp_db_path = "energyiea.db"
            
            with open(temp_db_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            st.success("Archivo subido correctamente.")
            
            conn = sqlite3.connect(temp_db_path)
            
            #df = pd.read_sql_query("SELECT * FROM energy_view_final", conn)

            df_Country = pd.read_sql("SELECT * FROM Country", conn)
            df_Balance = pd.read_sql("SELECT * FROM Balance", conn)
            df_Product = pd.read_sql("SELECT * FROM Product", conn)
            
            df = pd.read_sql("SELECT * FROM EnergyIEA", conn)
            
            df['Time'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
            
            st.write("Cabecera de la Tabla - EnergyIEA")
            
            st.dataframe(df.head(5))
            
            st.write(f"Shape: {df.shape}")
            
            st.write("Cabecera de la Tabla - Country")
            
            st.dataframe(df_Country.head(5))
            
            st.write(f"Shape: {df_Country.shape}")
            
            st.write("Cabecera de la Tabla - Balance")
            
            st.dataframe(df_Balance.head(5))
            
            st.write(f"Shape: {df_Balance.shape}")
            
            st.write("Cabecera de la Tabla - Product")
            
            st.dataframe(df_Product.head(5))
            
            st.write(f"Shape: {df_Product.shape}")
            
            st.write("Descripción - Variable Value (Todos los Registros)")
            
            st.dataframe(df['Value'].describe())
            
            st.write("Porcentaje de Valores Nulos")
            
            st.dataframe(df.isnull().sum()/len(df)*100)
            
            st.write(f"Cantidad de Valores Duplicados: {df.duplicated().sum()}")
            
            st.warning("Aplicamos la Operación: 'df.dropna(subset=['Value'], inplace=True)' para eliminar los nulos.")
            
            #st.write("Aplicamos la Operación: 'df.dropna(subset=['Value'], inplace=True)' para eliminar los nulos.")
            
            df.dropna(subset=['Value'], inplace=True)
            
            st.write("Porcentaje de Valores Nulos - Final")
            
            st.write(f"Shape: {df.shape}")
            
            st.dataframe(df.isnull().sum()/len(df)*100)
            
            st.write("Descripción - Variable Value (Luego de Eliminar Nulos)")
            
            st.dataframe(df['Value'].describe())
            
            #st.write("Gráficos para el Análisis Descriptivo - Variables Categóricas")
            st.header("Gráficos para el Análisis Descriptivo - Variables Categóricas")
            # Supongamos que ya tienes tus DataFrames: df, df_product, df_country, df_balance

            # 1. Unir los DataFrames con sus respectivas tablas de nombres
            df_Product_Merged = pd.merge(df, df_Product, on='Product_ID', how='left')
            df_Country_Merged = pd.merge(df, df_Country, on='Country_ID', how='left')
            df_Balance_Merged = pd.merge(df, df_Balance, on='Balance_ID', how='left')

            # 2. Calcular las frecuencias de cada categoría
            product_counts = df_Product_Merged['Product'].value_counts()   # Frecuencia de productos
            country_counts = df_Country_Merged['Country'].value_counts()   # Frecuencia de países
            balance_counts = df_Balance_Merged['Balance'].value_counts()   # Frecuencia de balances
            year_counts = df['Year'].value_counts().sort_index()           # Frecuencia de años
            month_counts = df['Month'].value_counts().sort_index()         # Frecuencia de meses

            # Gráfico individual: Frecuencia de Productos
            fig_product = go.Figure(
                data=go.Bar(
                    x=product_counts.index,
                    y=product_counts.values,
                    marker_color=get_colors(product_counts)
                )
            )
            fig_product.update_layout(
                title="Frecuencia de Productos", 
                xaxis_title="Producto", 
                yaxis_title="Frecuencia", 
                xaxis_tickangle=320,
            )
            
            st.plotly_chart(fig_product, use_container_width=True)

            # Gráfico individual: Frecuencia de Países
            fig_country = go.Figure(
                data=go.Bar(
                    x=country_counts.index,
                    y=country_counts.values,
                    marker_color=get_colors(country_counts)
                )
            )
            fig_country.update_layout(title="Frecuencia de Países", xaxis_title="País", yaxis_title="Frecuencia", xaxis_tickangle=270)
            st.plotly_chart(fig_country, use_container_width=True)

            # Gráfico individual: Frecuencia de Balances
            fig_balance = go.Figure(
                data=go.Bar(
                    x=balance_counts.index,
                    y=balance_counts.values,
                    marker_color=get_colors(balance_counts)
                )
            )
            fig_balance.update_layout(title="Frecuencia de Balances", xaxis_title="Balance", yaxis_title="Frecuencia")
            st.plotly_chart(fig_balance, use_container_width=True)

            # Gráfico individual: Frecuencia de Años
            fig_year = go.Figure(
                data=go.Bar(
                    x=year_counts.index,
                    y=year_counts.values,
                    marker_color=get_colors(year_counts)
                )
            )
            fig_year.update_layout(title="Frecuencia de Años", xaxis_title="Año", yaxis_title="Frecuencia")
            st.plotly_chart(fig_year, use_container_width=True)

            # Gráfico individual: Frecuencia de Meses
            fig_month = go.Figure(
                data=go.Bar(
                    x=month_counts.index,
                    y=month_counts.values,
                    marker_color=get_colors(month_counts)
                )
            )
            fig_month.update_layout(title="Frecuencia de Meses", xaxis_title="Mes", yaxis_title="Frecuencia")
            st.plotly_chart(fig_month, use_container_width=True)

        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            
def vista_analisis_exploratorio():
    """
    Crea un mapa dinámico con filtros interactivos.
    """

    st.header("Análisis Exploratorio - Dataset Month Electricity Statistics (IEA)")

    # Cargar datos desde SQLite
    uploaded_file = st.file_uploader("Elige un archivo de Base de Datos (Sqlite3)", type=["db", "sqlite3"])
    if uploaded_file:
        try:
            temp_db_path = "energyiea.db"
            
            with open(temp_db_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            st.success("Archivo subido correctamente.")
            
            conn = sqlite3.connect(temp_db_path)
            
            #df = pd.read_sql_query("SELECT * FROM energy_view_final", conn)

            df_Country = pd.read_sql("SELECT * FROM Country", conn)
            df_Balance = pd.read_sql("SELECT * FROM Balance", conn)
            df_Product = pd.read_sql("SELECT * FROM Product", conn)
            
            df = pd.read_sql("SELECT * FROM EnergyIEA", conn)
            
            df.dropna(subset=['Value'], inplace=True)
            
            st.markdown("""
            ## **Análisis de Energías Renovables: ¿Cuál tiene el mayor margen de crecimiento?**

            ---

            ### **Pregunta Central**
            **¿Cuál energía renovable tiene el mayor margen de crecimiento?**

            Para responder esta pregunta, analizaremos datos históricos de producción de energías renovables, calcularemos su crecimiento y proyectaremos su potencial futuro.

            ---
            """)
            
            
            df_merged = pd.merge(df, df_Product, left_on='Product_ID', right_on='Product_ID', how='left')

            renewable_products = ['Total Renewables (Hydro, Geo, Solar, Wind, Other)']#'Hydro', 'Combustible Renewables', 'Solar', 'Wind', 'Geothermal']

            df_renewables = df_merged[df_merged['Product'].isin(renewable_products)]
            #con Total Renewables (Hydro, Geo, Solar, Wind, Other)


            df_renewables_with_country = pd.merge(
                df_renewables, 
                df_Country, 
                left_on='Country_ID', 
                right_on='Country_ID', 
                #solo quiero las columnas country y product y value
                how='left',
            )
            df_renewables_by_country = df_renewables_with_country.groupby('Country')['Value'].sum().reset_index()
            df_renewables_by_country.columns = ['Country', 'Total_Production']


            fig = px.choropleth(
                df_renewables_by_country,
                locations="Country",
                locationmode="country names",
                color="Total_Production",
                color_continuous_scale=px.colors.sequential.Blackbody,  # Escala de colores black
                title="Producción Total de Energías Renovables por País",
                labels={"Total_Production": "Producción Total (GWh)"},
            )

            # Personalizar el mapa
            fig.update_layout(
                geo=dict(
                    showframe=False,
                    showcoastlines=True,   
                    projection_type="equirectangular",  

                ),
                margin={"r": 0, "t": 40, "l": 0, "b": 0},  # Márgenes
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(r"""
            ---
            
            **CAGR (Tasa de Crecimiento Anual Compuesto)**
            El **CAGR** es una métrica que mide la tasa de crecimiento anual promedio de una variable (en este caso, la producción de energía) durante un período de tiempo. Se calcula como:

            $$
            \text{CAGR} = \left(\frac{\text{Valor Final}}{\text{Valor Inicial}}\right)^{\frac{1}{n}} - 1
            $$

            Donde:
            - **Valor Final**: Producción en el último año.
            - **Valor Inicial**: Producción en el primer año.
            - **n**: Número de años.

            ---
            """)
            
            # Crear df_energy_renewable_countries_años con el total de producción donde coinciden el país y el año
            df_energy_renewable_countries_años = df_renewables_with_country.groupby(['Country', 'Year'])['Value'].sum().reset_index()
            df_energy_renewable_countries_años.columns = ['Country', 'Year', 'Total_Production']
            
            st.dataframe(df_energy_renewable_countries_años.head(5))
            #df_energy_renewable_countries_años.head(100)

            # Filtrar los datos para los años 2019 y 2024
            df_tasa_crecimiento = df_energy_renewable_countries_años[df_energy_renewable_countries_años['Year'].isin([2019, 2024])]
            
            st.dataframe(df_tasa_crecimiento.head(5))
            #df_tasa_crecimiento.head(100)

            #calcular tasa de crecimiento de cada país
            df_CAGR = df_tasa_crecimiento.pivot(index='Country', columns='Year', values='Total_Production').reset_index()
            df_CAGR['CAGR'] = ((df_CAGR[2024] / df_CAGR[2019]) ** (1/11) - 1) * 100
            df_CAGR = df_CAGR[['Country', 'CAGR']].sort_values('CAGR', ascending=False)
            
            st.dataframe(df_CAGR.head(5))
            #df_CAGR.head(10)
            
            # Crear el gráfico de barras horizontal usando Plotly Express
            fig = px.bar(
                df_CAGR.head(54),
                x='CAGR',
                y='Country',
                orientation='h',  # Para que las barras sean horizontales
                color='Country',  # Asigna colores distintos a cada país
                color_continuous_scale='viridis'  # Paleta de colores
            )

            # Actualizar el layout para agregar títulos y eliminar la leyenda
            fig.update_layout(
                title="Top 10 Países por CAGR energias renovables",
                xaxis_title="CAGR (%)",
                yaxis_title="País",
                showlegend=False,
                height=900,  # Ajusta la altura según tus necesidades
                width=800    # Ajusta el ancho según tus necesidades
            )

            # Mostrar el gráfico en Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            df_energy_renewable_countries_años = df_renewables_with_country.groupby(['Country', 'Year'])['Value'].sum().reset_index()
            df_energy_renewable_countries_años.columns = ['Country', 'Year', 'Total_Production']
            #df_energy_renewable_countries_años.head(100)
            df_tasa_crecimiento = df_energy_renewable_countries_años[df_energy_renewable_countries_años['Year'].isin([2019, 2024])]
            df_CAGR = df_tasa_crecimiento.pivot(index='Country', columns='Year', values='Total_Production').reset_index()
            df_CAGR['CAGR'] = ((df_CAGR[2024] / df_CAGR[2019]) ** (1/11) - 1) * 100
            df_CAGR = df_CAGR[['Country', 'CAGR']].sort_values('CAGR', ascending=False)
            #df_CAGR.head(10)
            renewable_energy = [
                "Hydro",
                "Combustible Renewables",
                "Solar",
                "Wind",
                "Geothermal",
                "Other Renewables"
            ]

            df_energias_renovables = pd.merge(df, df_Product, left_on='Product_ID', right_on='Product_ID', how='left')
            df_energias_renovables = df_energias_renovables[df_energias_renovables['Product'].isin(renewable_energy)]
            #df_energias_renovables.head(20)
            df_energias_renovables_años = df_energias_renovables.groupby(['Product', 'Year'])['Value'].sum().reset_index()
            df_energias_renovables_años.columns = ['Product', 'Year', 'Total_Production']
            #df_energias_renovables_años.head(100)


            df_cagr = df_energias_renovables_años.pivot(index='Product', columns='Year', values='Total_Production').reset_index()

            df_cagr['CAGR'] = ((df_cagr[2024] / df_cagr[2019]) ** (1/11) - 1) * 100

            df_cagr = df_cagr[['Product', 'CAGR']].sort_values('CAGR', ascending=False)
            
            # Creación del gráfico de barras horizontal
            fig = px.bar(
                df_cagr,
                x='CAGR',
                y='Product',
                color='Product',  # Equivale a hue='Product' en Seaborn
                orientation='h',  # Barras horizontales
                color_discrete_sequence=px.colors.sequential.Viridis,
                title='Energías Renovables y Tasa de Crecimiento Anual Compuesto (CAGR)'
            )

            # Ajustes de layout
            fig.update_layout(
                xaxis_title='CAGR (%)',
                yaxis_title='Energía Renovable',
                showlegend=False,  # Equivale a legend=False en Seaborn
                template='plotly_white'
            )

            # Mostrar en Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            df_Solar_Colombia = pd.merge(df, df_Product, left_on='Product_ID', right_on='Product_ID', how='left')
            df_Solar_Colombia = pd.merge(df_Solar_Colombia, df_Country, left_on='Country_ID', right_on='Country_ID', how='left')
            df_Solar_Colombia = df_Solar_Colombia[(df_Solar_Colombia['Product_ID'] == 7) & (df_Solar_Colombia['Country_ID'] == 5)]
            
            st.dataframe(df_Solar_Colombia.head(15))
            
            grafico_Frecuencia=df_energy_renewable_countries_años[df_energy_renewable_countries_años['Country']=='Colombia']
            
            # Suponiendo que 'grafico_frecuencia' es tu DataFrame con las columnas 'Year' y 'Total_Production'
            fig = px.line(
                grafico_Frecuencia,
                x='Year',
                y='Total_Production',
                title='Producción de Energías Renovables en Colombia',
                markers=True,  # Agrega marcadores en cada punto
                labels={'Year': 'Año', 'Total_Production': 'Producción Total (GWh)'}
            )

            # Ajustar tamaño y configurar la grilla
            fig.update_layout(
                width=1200,
                height=600,
                xaxis_title='Año',
                yaxis_title='Producción Total (GWh)',
                template='plotly_white'
            )
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)

            # Mostrar el gráfico en Streamlit
            st.plotly_chart(fig, use_container_width=True)

            # Agrupamos por 'Year' y calculamos la media y la desviación estándar
            df_grouped = df_Solar_Colombia.groupby('Year')['Value'].agg(['mean', 'std']).reset_index()

            # Definimos la banda superior e inferior (aquí usamos ±1 std como ejemplo)
            df_grouped['upper'] = df_grouped['mean'] + df_grouped['std']
            df_grouped['lower'] = df_grouped['mean'] - df_grouped['std']

            fig_ci = go.Figure()

            # --- Traza 1: Banda superior (invisible) ---
            fig_ci.add_trace(
                go.Scatter(
                    x=df_grouped['Year'],
                    y=df_grouped['upper'],
                    mode='lines',
                    line_color='rgba(0, 0, 0, 0)',  # línea transparente
                    showlegend=False,
                    name='upper'
                )
            )

            # --- Traza 2: Banda inferior con relleno hasta la traza anterior ---
            fig_ci.add_trace(
                go.Scatter(
                    x=df_grouped['Year'],
                    y=df_grouped['lower'],
                    mode='lines',
                    fill='tonexty',                  # rellena el área hasta la traza anterior (upper)
                    fillcolor='rgba(0, 0, 255, 0.2)',# color semitransparente para la banda
                    line_color='rgba(0, 0, 0, 0)',   # línea transparente
                    showlegend=False,
                    name='lower'
                )
            )

            # --- Traza 3: Línea principal (media) ---
            fig_ci.add_trace(
                go.Scatter(
                    x=df_grouped['Year'],
                    y=df_grouped['mean'],
                    mode='lines+markers',
                    line=dict(color='blue', width=3),
                    marker=dict(size=6),
                    name='Media'
                )
            )


            fig_ci.update_layout(
                title="Producción de Energía Solar en Colombia (Intervalo de Confianza)",
                xaxis_title="Año",
                yaxis_title="Producción (GWh)",
                template='seaborn',  # Usa la plantilla 'seaborn' para colores y estilo
                width=800,
                height=600
            )

            st.plotly_chart(fig_ci, use_container_width=True)


            # Segundo gráfico: Producción de Energía Solar en Colombia por cada mes del año
            # Definimos una paleta personalizada para los distintos años
            custom_palette = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'pink', 'yellow', 'gray', 'black']

            fig_month = px.line(
                df_Solar_Colombia,
                x='Month',
                y='Value',
                color='Year',  # Separa la línea por cada año
                title="Producción de Energía Solar en Colombia por cada mes del año",
                labels={"Month": "Mes", "Value": "Producción (GWh)"},
                color_discrete_sequence=custom_palette
            )
            fig_month.update_layout(width=800, height=600)
            st.plotly_chart(fig_month, use_container_width=True)
            
            
            st.markdown("""
            > ## **Conclusión General**

            Al analizar los datos de producción de energía solar en Colombia, se observó un crecimiento extraordinario durante el mes Octubre de 2024, cuando la generación se duplicó de 2,095 Millones de GWh en Enero del 2024 a 4,425 Millones de GWh en Octubre del 2024 (un aumento de más del 100%). Este salto se puede explicar por la confluencia de varios factores:

            - **Políticas de fomento e incentivos regulatorios:**  
            Las medidas gubernamentales han facilitado la agilización de trámites y la obtención de permisos, lo que ha permitido el cierre financiero de nuevos proyectos solares. Estas políticas, junto con incentivos fiscales, han incentivado a los grandes actores del sector a invertir en tecnología fotovoltaica.

            - **Inversiones estratégicas y desarrollo de infraestructura:**  
            Empresas como Enel Colombia han impulsado proyectos a gran escala (por ejemplo, los parques solares La Loma, Guayepo y El Paso) que han contribuido significativamente a la capacidad instalada del país. Estas inversiones han cuadruplicado la capacidad instalada renovable no convencional respecto al año anterior.

            - **Avances tecnológicos:**  
            La mejora en la eficiencia de los paneles solares y la implementación de tecnologías innovadoras han permitido aumentar la producción energética, haciendo que la energía solar sea cada vez más competitiva frente a otras fuentes de energía.

            En conjunto, estos factores han permitido que la producción de energía solar en Colombia experimente un crecimiento notable en un corto periodo, destacando la importancia de continuar fortaleciendo las políticas de apoyo y la inversión en renovables para consolidar una matriz energética más limpia y sostenible.

            - En abril se inaguraron 7 plantas
            # Referencias

            - **El Colombiano**. *Colombia cerró 2024 con 33 nuevas plantas solares en operación, la mitad de lo proyectado*. Recuperado de [El Colombiano](https://www.elcolombiano.com/negocios/energias-renovables-los-proyectos-que-entraron-en-operacion-en-colombia-en-2024-NB26278005)
            - **Review Energy**. *¿Cuáles son las expectativas renovables para Colombia en el 2024?*. Recuperado de [Review Energy](https://www.review-energy.com/otras-fuentes/-cuales-son-las-expectativas-renovables-para-colombia-en-el-2024)            
            """)
                        
            

        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            
            
def vista_renovables_por_pais():
    """
    Crea un mapa dinámico con filtros interactivos.
    """

    st.header("Total Anual de Energías Renovables por País - Dataset Month Electricity Statistics (IEA)")

    # Cargar datos desde SQLite
    uploaded_file = st.file_uploader("Elige un archivo de Base de Datos (Sqlite3)", type=["db", "sqlite3"])
    if uploaded_file:
        try:
            temp_db_path = "energyiea.db"
            
            with open(temp_db_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            st.success("Archivo subido correctamente.")
            
            conn = sqlite3.connect(temp_db_path)
            
            #df = pd.read_sql_query("SELECT * FROM energy_view_final", conn)

            df_Country = pd.read_sql("SELECT * FROM Country", conn)
            df_Balance = pd.read_sql("SELECT * FROM Balance", conn)
            df_Product = pd.read_sql("SELECT * FROM Product", conn)
            
            df = pd.read_sql("SELECT * FROM EnergyIEA", conn)
            
            df.dropna(subset=['Value'], inplace=True)  
            
            df_merged = pd.merge(df, df_Product, left_on='Product_ID', right_on='Product_ID', how='left')

            renewable_products = ['Total Renewables (Hydro, Geo, Solar, Wind, Other)']#'Hydro', 'Combustible Renewables', 'Solar', 'Wind', 'Geothermal']

            df_renewables = df_merged[df_merged['Product'].isin(renewable_products)]
            #con Total Renewables (Hydro, Geo, Solar, Wind, Other)


            df_renewables_with_country = pd.merge(
                df_renewables, 
                df_Country, 
                left_on='Country_ID', 
                right_on='Country_ID', 
                #solo quiero las columnas country y product y value
                how='left',
            )
            df_renewables_by_country = df_renewables_with_country.groupby('Country')['Value'].sum().reset_index()
            df_renewables_by_country.columns = ['Country', 'Total_Production']
            
            # Crear df_energy_renewable_countries_años con el total de producción donde coinciden el país y el año
            df_energy_renewable_countries_años = df_renewables_with_country.groupby(['Country', 'Year'])['Value'].sum().reset_index()
            df_energy_renewable_countries_años.columns = ['Country', 'Year', 'Total_Production']
            
            st.dataframe(df_energy_renewable_countries_años.head(5))
            #df_energy_renewable_countries_años.head(100)

            # Filtrar los datos para los años 2019 y 2024
            df_tasa_crecimiento = df_energy_renewable_countries_años[df_energy_renewable_countries_años['Year'].isin([2019, 2024])]
            
            #st.dataframe(df_tasa_crecimiento.head(5))
            #df_tasa_crecimiento.head(100)

            #calcular tasa de crecimiento de cada país
            df_CAGR = df_tasa_crecimiento.pivot(index='Country', columns='Year', values='Total_Production').reset_index()
            df_CAGR['CAGR'] = ((df_CAGR[2024] / df_CAGR[2019]) ** (1/11) - 1) * 100
            df_CAGR = df_CAGR[['Country', 'CAGR']].sort_values('CAGR', ascending=False)
            
            #st.dataframe(df_CAGR.head(5))
            #df_CAGR.head(10)
            
            df_Solar_Colombia = pd.merge(df, df_Product, left_on='Product_ID', right_on='Product_ID', how='left')
            df_Solar_Colombia = pd.merge(df_Solar_Colombia, df_Country, left_on='Country_ID', right_on='Country_ID', how='left')
            df_Solar_Colombia = df_Solar_Colombia[(df_Solar_Colombia['Product_ID'] == 7) & (df_Solar_Colombia['Country_ID'] == 5)]
            
            #st.dataframe(df_Solar_Colombia.head(15))

            # Agrupamos por 'Year' y calculamos la media y la desviación estándar
            df_grouped = df_Solar_Colombia.groupby('Year')['Value'].agg(['mean', 'std']).reset_index()

            # Definimos la banda superior e inferior (aquí usamos ±1 std como ejemplo)
            df_grouped['upper'] = df_grouped['mean'] + df_grouped['std']
            df_grouped['lower'] = df_grouped['mean'] - df_grouped['std']
            
            
            # 1. Selección múltiple de países
            all_countries = df_energy_renewable_countries_años['Country'].unique().tolist()
            selected_countries = st.multiselect(
                "Selecciona los países que deseas visualizar:",
                options=all_countries,
                default=["Colombia"]  # Puedes poner un valor por defecto
            )

            # 2. Filtrar el DataFrame según los países seleccionados
            df_filtered = df_energy_renewable_countries_años[
                df_energy_renewable_countries_años['Country'].isin(selected_countries)
            ]

            # 3. Crear el gráfico de líneas con color='Country'
            fig = px.line(
                df_filtered,
                x='Year',
                y='Total_Production',
                color='Country',  # Diferencia por país
                markers=True,
                title='Producción de Energías Renovables por País',
                labels={'Year': 'Año', 'Total_Production': 'Producción Total (GWh)'}
            )

            # 4. Ajustar la apariencia del gráfico
            fig.update_layout(
                width=1200,
                height=600,
                xaxis_title='Año',
                yaxis_title='Producción Total (GWh)',
                template='plotly_white'
            )
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)

            # 5. Mostrar el gráfico en Streamlit
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")

# Menú de navegación en la barra lateral
st.sidebar.title("Navegación - Bootcamp Data Analysis")

pagina = st.sidebar.selectbox("Menú de Páginas", 
                              ["Análisis Descriptivo", "Análisis Exploratorio", "Predicción", "Series de Tiempo", "Total Anual de Energías Renovables por País"])

# Agrega enlaces a LinkedIn con íconos
st.sidebar.markdown(
    """
    <a href="https://www.linkedin.com/in/daniel-vergara-nieto/" target="_blank" style="text-decoration:none;">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20" style="vertical-align:middle; margin-right:5px;" />
        Juan Daniel Vergara
    </a>
    <br><br>
    <a href="https://www.linkedin.com/in/andres-felipe-lemus-v-7943882a9/" target="_blank" style="text-decoration:none;">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20" style="vertical-align:middle; margin-right:5px;" />
        Andres Felipe Lemus
    </a>
    """,
    unsafe_allow_html=True
)


if pagina == "Predicción":
    vista_prediccion()
elif pagina == "Series de Tiempo":
    vista_graficos()
elif pagina == "Análisis Descriptivo":
    vista_analisis_descriptivo()
elif pagina == "Análisis Exploratorio":
    vista_analisis_exploratorio()
elif pagina == "Total Anual de Energías Renovables por País":
    vista_renovables_por_pais()
