
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, QuantileRegressor
from sklearn.metrics import mean_squared_error, mean_pinball_loss
from sklearn.datasets import fetch_california_housing
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

# =============================================================================
#   ESTILO VISUAL PARA CLI
# =============================================================================
# Colores base para las gráficas
bg_color = '#383838'
text_color = '#FFFFFF'
plot_bg_color = '#000000'

class Colors:
    """Clase para guardar los códigos de color para la salida en la terminal."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(title):
    """Imprime un encabezado principal estilizado."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}📊 {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")

def print_subheader(title):
    """Imprime un subencabezado estilizado."""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}🔹 {title}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'-'*40}{Colors.ENDC}")

def print_metric(name, value, color=Colors.OKGREEN):
    """Imprime una métrica formateada."""
    if isinstance(value, float):
        print(f"  {color}• {name}: {value:.4f}{Colors.ENDC}")
    else:
        print(f"  {color}• {name}: {value}{Colors.ENDC}")

def print_info(message, color=Colors.OKBLUE):
    """Imprime un mensaje informativo."""
    print(f"{color}ℹ️  {message}{Colors.ENDC}")

def print_success(message):
    """Imprime un mensaje de éxito."""
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")

# =============================================================================
#         PREDICCIÓN CON ELASTIC NET
# =============================================================================
def run_elastic_net():
    """
    Carga los datos de California Housing, entrena un modelo Elastic Net,
    lo evalúa y visualiza los coeficientes más importantes.
    """
    print_header("1. Elastic Net: Predicción de Precios de Viviendas")

    # Cargar y preparar el dataset
    print_info("Cargando y preparando el dataset de California Housing...")
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar los datos es importante para Elastic Net
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print_success("Datos preparados y divididos.")

    # Aplicar Elastic Net
    print_info("Ajustando el modelo Elastic Net...")
    # alpha controla la regularización total, l1_ratio la mezcla entre L1 y L2
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    elastic_net.fit(X_train_scaled, y_train)
    print_success("Modelo Elastic Net entrenado.")

    # Evaluar el modelo
    print_subheader("Evaluación del Modelo")
    y_pred = elastic_net.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print_metric("Error Cuadrático Medio (RMSE)", rmse)

    # Analizar coeficientes
    print_subheader("Análisis de Coeficientes")
    print_info("Elastic Net ayuda a seleccionar las características más relevantes.")
    coefficients = pd.Series(elastic_net.coef_, index=housing.feature_names).sort_values(ascending=False)
    print(coefficients)

    # Visualizar resultados
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
    coefficients.plot(kind='bar', ax=ax)
    ax.set_title('Importancia de las Características (Coeficientes de Elastic Net)', color=text_color)
    ax.set_ylabel('Valor del Coeficiente', color=text_color)
    ax.set_facecolor(plot_bg_color)
    ax.tick_params(axis='x', colors=text_color, rotation=45)
    ax.tick_params(axis='y', colors=text_color)
    ax.spines['left'].set_color(text_color)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['top'].set_color(bg_color)
    ax.spines['right'].set_color(bg_color)
    fig.tight_layout()
    plt.show()

# =============================================================================
#         REGRESIÓN CUANTÍLICA
# =============================================================================
def run_quantile_regression():
    """
    Carga el dataset 'Adult', entrena modelos de Regresión Cuantílica
    para varios percentiles y visualiza las predicciones.
    """
    print_header("2. Regresión Cuantílica: Estimación de Ingresos")

    # Cargar y preparar el dataset
    print_info("Cargando y preparando el dataset Adult...")
    # URL directa para evitar problemas con fetch_openml
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    # Usamos solo algunas columnas para simplificar
    adult_df = pd.read_csv(url, header=None, names=columns, na_values=' ?', skipinitialspace=True)
    adult_df.dropna(inplace=True)

    # Para este ejemplo, predeciremos 'hours-per-week' basado en 'age'
    X = adult_df[['age']].values
    y = adult_df['hours-per-week'].values
    print_success("Datos preparados.")

    # Aplicar Regresión Cuantílica para los percentiles 10, 50 y 90
    quantiles = [0.1, 0.5, 0.9]
    predictions = {}

    print_subheader("Ajuste y Evaluación de Modelos")
    for q in quantiles:
        print_info(f"Ajustando modelo para el cuantil {q*100:.0f}...")
        quantile_reg = QuantileRegressor(quantile=q, alpha=0) # alpha=0 es una regresión sin regularización
        quantile_reg.fit(X, y)

        # Guardamos la predicción para la visualización
        # Creamos una línea de puntos para graficar la predicción
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        predictions[q] = quantile_reg.predict(x_range)

        # Evaluar con la pérdida Pinball
        y_pred = quantile_reg.predict(X)
        pinball_loss = mean_pinball_loss(y, y_pred, alpha=q)
        print_metric(f'Pérdida Pinball (q={q})', pinball_loss)

    # Visualizar resultados
    print_info("Generando visualización de las regresiones cuantílicas...")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
    ax.scatter(X, y, alpha=0.1, color='cyan', label='Datos reales')

    for q, y_pred_line in predictions.items():
        ax.plot(x_range, y_pred_line, label=f'Cuantil {q*100:.0f}', linewidth=2)

    ax.legend()
    ax.set_title('Regresión Cuantílica: Horas por Semana vs. Edad', color=text_color)
    ax.set_xlabel('Edad', color=text_color)
    ax.set_ylabel('Horas por Semana', color=text_color)
    ax.set_facecolor(plot_bg_color)
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    ax.spines['left'].set_color(text_color)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['top'].set_color(bg_color)
    ax.spines['right'].set_color(bg_color)
    ax.legend(facecolor='grey', labelcolor=text_color)
    fig.tight_layout()
    plt.show()

# =============================================================================
#         MODELO DE AUTORREGRESIÓN VECTORIAL (VAR)
# =============================================================================
def run_var_model():
    """
    Carga datos macroeconómicos, ajusta un modelo VAR, selecciona el lag
    óptimo y realiza una predicción a futuro.
    """
    print_header("3. VAR: Proyección de Indicadores Económicos")

    # Cargar y preparar el dataset
    print_info("Cargando y preparando el dataset macroeconómico...")
    data = sm.datasets.macrodata.load_pandas().data
    # Seleccionamos las variables de interés
    df_var = data[['realgdp', 'realcons', 'realinv']]

    # Los modelos VAR requieren que las series sean estacionarias.
    # Una forma común de lograrlo es diferenciando las series.
    df_diff = df_var.diff().dropna()
    print_success("Datos preparados y diferenciados para asegurar estacionariedad.")

    # Dividir en entrenamiento y prueba (los últimos 5 puntos para validar)
    train_data = df_diff[:-5]
    test_data = df_diff[-5:]

    # Aplicar VAR: Seleccionar lag óptimo
    print_subheader("Selección del Lag Óptimo")
    model_selection = VAR(train_data)
    lag_selection_results = model_selection.select_order(maxlags=10)
    selected_lag = lag_selection_results.aic
    print_metric("Lag óptimo seleccionado (AIC)", selected_lag)

    # Entrenar el modelo con el lag óptimo
    print_info(f"Entrenando modelo VAR con {selected_lag} lags...")
    var_model = VAR(train_data)
    var_results = var_model.fit(selected_lag)
    print_success("Modelo VAR entrenado.")

    # Predecir 5 pasos hacia adelante
    print_subheader("Predicción a 5 Pasos")
    forecast = var_results.forecast(train_data.values, steps=5)
    df_forecast = pd.DataFrame(forecast, index=test_data.index, columns=df_diff.columns)
    print(df_forecast)

    # Calcular errores de VAR
    print_subheader("Evaluación de Errores del Modelo VAR")
    for col in df_diff.columns:
        rmse_var = np.sqrt(mean_squared_error(test_data[col], df_forecast[col]))
        print_metric(f"RMSE VAR – {col}", rmse_var)

    # Visualizar los resultados y la predicción
    print_info("Generando visualización de las predicciones del modelo VAR...")
    for col in df_diff.columns:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
        ax.plot(train_data.index, train_data[col], label='Datos de Entrenamiento', color='cyan')
        ax.plot(test_data.index, test_data[col], label='Valores Reales (Test)', color='lime', marker='o')
        ax.plot(df_forecast.index, df_forecast[col], label='Predicción VAR', color='magenta', linestyle='--')

        ax.set_title(f'Predicción de {col}', color=text_color)
        ax.set_ylabel('Valor (Diferenciado)', color=text_color)
        ax.legend()
        ax.set_facecolor(plot_bg_color)
        ax.tick_params(axis='x', colors=text_color, rotation=15)
        ax.tick_params(axis='y', colors=text_color)
        ax.spines['left'].set_color(text_color)
        ax.spines['bottom'].set_color(text_color)
        ax.spines['top'].set_color(bg_color)
        ax.spines['right'].set_color(bg_color)
        ax.legend(facecolor='grey', labelcolor=text_color)
        fig.tight_layout()
        plt.show()

# =============================================================================
#        EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    run_elastic_net()
    run_quantile_regression()
    run_var_model()
    print_header("Actividad Completada")
