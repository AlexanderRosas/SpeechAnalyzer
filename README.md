# 🎤 Speech Analyzer AI - Asistente de Oratoria Inteligente

> Un sistema de análisis en tiempo real para entrenamiento de oratoria y debate, utilizando Visión por Computadora y Procesamiento de Lenguaje Natural (NLP) offline.

![Estado del Proyecto](https://img.shields.io/badge/Estado-Prototipo%20Funcional-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Librerías](https://img.shields.io/badge/Libs-OpenCV%20|%20MediaPipe%20|%20Vosk-orange)

## 📋 Descripción

**Speech Analyzer AI** es una herramienta diseñada para ayudar a estudiantes, oradores y debatientes a mejorar su comunicación no verbal y la coherencia de su discurso. 

El sistema utiliza **Fusión de Sensores (Sensor Fusion)** para cruzar datos visuales (gestos faciales y corporales) con datos auditivos (análisis de sentimiento del texto hablado) en tiempo real. El objetivo es detectar la **Congruencia Emocional**: ¿Coincide lo que dices con la cara que pones?

## 🚀 Características Principales

- **👁️ Análisis Facial en Tiempo Real:** Detección de sonrisas, ceño fruncido, apertura de boca y gestos corporales (manos levantadas) usando *MediaPipe*.
- **🗣️ Transcripción Offline:** Uso de la librería *Vosk* para transcripción de voz a texto sin necesidad de internet y con baja latencia.
- **🧠 Detección de Congruencia:** Algoritmo lógico que compara el sentimiento del texto (Positivo/Negativo) con la expresión facial para alertar incongruencias (ej. decir algo triste sonriendo).
- **📊 Dashboard Visual:** Interfaz gráfica construida con OpenCV que muestra métricas, semáforo de coherencia y transcripción en vivo.
- **💾 Registro de Sesiones:** Exportación automática de datos a CSV para análisis posterior.

## 🛠️ Instalación y Configuración

Sigue estos pasos para ejecutar el proyecto en tu entorno local.

### Prerrequisitos
- Python 3.8 o superior.
- Webcam y Micrófono funcionales.

### 1. Clonar el repositorio
```bash
git clone [https://github.com/AlexanderRosas/SpeechAnalyzer)
cd SpeechAnalyzer
```

### 2. Crear Entorno Virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar Modelo de Voz
1. Este proyecto usa Vosk (offline). Debes descargar el modelo manualmente:
2. Ve a Vosk Models.
3. Descarga el modelo vosk-model-small-es-0.42 (o la versión en español que prefieras).
4. Descomprime el archivo .zip.
5. Renombra la carpeta extraída simplemente a model.
6. Mueve la carpeta model a la raíz del proyecto (junto a main.py).

### 📂 Estructura del Proyecto
```PlainText
SpeechAnalyzer/
│
├── data/                  # CSVs generados automáticamente con los logs de la sesión
├── model/                 # Carpeta del modelo Vosk (Descargada manualmente)
│   ├── am/
│   ├── conf/
│   └── ...
├── venv/                  # Entorno virtual
├── main.py                # Código fuente principal (Lógica y GUI)
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Documentación
```

### 🖥️ Uso
Una vez configurado, ejecuta el script principal:
```PlainText
python main.py
```

### Controles
  El sistema abrirá dos ventanas: Dashboard (Video y Métricas) y Transcripción.
  Presiona la tecla q en cualquiera de las ventanas para detener la sesión y guardar el CSV.

### Interpretación del Dashboard
  Semáforo Verde (COHERENTE): Tu expresión facial coincide con el sentimiento de tus palabras.
  Semáforo Rojo (ALERTA - INCONGRUENCIA):
    - Caso A: Estás diciendo algo positivo con cara de enojo/preocupación.
    - Caso B: Estás diciendo algo negativo/triste mientras sonríes (nervios o sarcasmo).
    
### Licencia
Este proyecto es de uso académico y libre distribución.
