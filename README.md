
# Clasificación de Emociones en Texto usando Machine Learning y Transformers

Este proyecto tiene como objetivo desarrollar y comparar distintos modelos de clasificación de emociones a partir de respuestas textuales proporcionadas por estudiantes. Se implementan tanto algoritmos tradicionales con TF-IDF como modelos de lenguaje preentrenados (Transformers) para identificar emociones como Felicidad, Tristeza, Ira, Miedo, Disgusto y Sorpresa.

## 📁 Estructura del Proyecto

```
Proyecto_Emociones/
│
├── data/
│   ├── Nuevo_Dataset_Patrones_Emocionales.csv   # Dataset original crudo
│   └── Dataset_Emociones_Limpio.csv             # Dataset preprocesado y etiquetado
│
├── notebooks/
│   ├── 01_EDA_Preprocesamiento.ipynb            # Análisis exploratorio de datos y limpieza
│   ├── 02_Modelos_Tradicionales.ipynb           # Implementación de SVM, KNN, NB, RF con TF-IDF
│   ├── 03_Transformers_BETO_RoBERTuito.ipynb    # Fine-tuning y predicción usando BETO y RoBERTuito
│   ├── 04_Embeddings_y_Clasificación.ipynb      # Extracción de embeddings + ML tradicionales
│   └── 05_Modelo_MarIA.ipynb                     # Entrenamiento del modelo MarIA
│
├── results/
│   ├── figuras/                                 # Gráficas comparativas generadas
│   ├── metrics/                                 # Reportes de clasificación y métricas por modelo
│   └── tablas/                                  # Tablas resumen para el documento
│
├── models/
│   └── transformers_checkpoints/                # Pesos y logs de modelos fine-tuneados
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

1. **Preprocesamiento de Datos:** Ejecuta `01_EDA_Preprocesamiento.ipynb`
2. **Modelos Tradicionales:** Corre `02_Modelos_Tradicionales.ipynb` para TF-IDF + SVM, NB, etc.
3. **Modelos Transformers:** Usa `03_Transformers_BETO_RoBERTuito.ipynb`
4. **Embeddings + ML Tradicional:** Ejecuta `04_Embeddings_y_Clasificación.ipynb`
5. **Modelo MarIA:** Corre `05_Modelo_MarIA.ipynb`

## 🧠 Reproducción de Resultados

Cada notebook está dividido en secciones explicativas. Para reproducir los resultados:

- Cargar correctamente el dataset limpio (`Dataset_Emociones_Limpio.csv`) en la carpeta `data/`.
- Asegurarte de que las clases estén correctamente etiquetadas.
- Verifica que el entorno tenga los paquetes adecuados instalados.
- Se recomienda usar Google Colab para acceder a aceleradores como GPU.

## 📊 Salidas Esperadas

- Gráficas comparativas (accuracy, F1-score)
- Matrices de confusión
- Reportes de clasificación por modelo
- Tablas de resumen exportadas

---

Este repositorio ha sido estructurado para facilitar la lectura, ejecución y análisis de resultados. Las salidas se almacenan automáticamente en la carpeta `results/`.

