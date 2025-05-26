
# Clasificación de Emociones en Texto usando Machine Learning y Transformers

Este proyecto tiene como objetivo desarrollar y comparar distintos modelos de clasificación de emociones a partir de respuestas textuales proporcionadas por estudiantes. Se implementan tanto algoritmos tradicionales con TF-IDF como modelos de lenguaje preentrenados (Transformers) para identificar emociones como Felicidad, Tristeza, Ira, Miedo, Disgusto y Sorpresa.

## 📁 Estructura del Proyecto

```
mineria-analisis-de-emociones/
│
├── data/
│   ├── Nuevo_Dataset_Patrones_Emocionales.csv   # Dataset original crudo
│   └── Dataset_Emociones_Limpio.csv             # Dataset preprocesado y etiquetado
│
├── notebooks/
│   ├── Proyecto.ipynb                          # Notebook con todo el contenido de Python
│
├── results/
│   ├── figuras/                                 # Gráficas comparativas generadas
│   ├── metrics/                                 # Reportes de clasificación y métricas por modelo
│   └── tablas/                                  # Tablas resumen para el documento
│
├── utils/
│   └── preprocessing.py                         # Funciones auxiliares para limpieza y tokenización
│
├── README.md                                     # Descripción del proyecto y guía de ejecución
└── requirements.txt                              # Lista de dependencias Python del proyecto
```

## 🔧 Instalación de Dependencias

Puedes instalar las librerías necesarias utilizando:

```bash
pip install -r requirements.txt
```

O bien desde Google Colab:

```python
!pip install -q transformers==4.30.2 datasets scikit-learn nltk wordcloud
```

## 📌 Modelos Comparados

- **Tradicionales:** SVM, KNN, Naive Bayes, Random Forest (con TF-IDF)
- **Transformers:** BETO, RoBERTuito, MarIA
- **Embeddings:** Extraídos desde Transformers para modelos clásicos


## 🚀 Ejecución del Proyecto

Todo el flujo de trabajo se encuentra consolidado en el archivo:

```
Proyecto.ipynb
```

Este cuaderno contiene de forma secuencial y documentada todas las etapas del análisis, desde la carga y limpieza de datos, hasta la evaluación de modelos tradicionales y modelos basados en transformers.

### Etapas incluidas en el notebook:

1. **Análisis Exploratorio de Datos (EDA)**
2. **Preprocesamiento de texto para modelos tradicionales**
3. **Vectorización con TF-IDF**
4. **Entrenamiento y evaluación de modelos clásicos (SVM, NB, RF, KNN)**
5. **Fine-tuning de modelos transformers (BETO, RoBERTuito, MarIA)**
6. **Extracción de embeddings y clasificación con ML tradicional**
7. **Comparación de resultados y visualizaciones**

### Para ejecutarlo correctamente:

- Asegúrate de tener el archivo `Dataset_Emociones_Limpio.csv` cargado en la misma ruta del notebook o en `/content/drive/MyDrive/...`.
- Usa **Google Colab** si deseas aprovechar GPU para entrenamiento con transformers.
- Verifica que se hayan instalado las dependencias listadas en `requirements.txt`.


## 📊 Salidas Esperadas

- Gráficas comparativas (accuracy, F1-score)
- Matrices de confusión
- Reportes de clasificación por modelo
- Tablas de resumen exportadas

---

Este repositorio ha sido estructurado para facilitar la lectura, ejecución y análisis de resultados. Las salidas se almacenan automáticamente en la carpeta `results/`.

