# Clasificación de Emociones en Texto usando Machine Learning y Transformers

Este proyecto tiene como objetivo desarrollar y comparar distintos modelos de clasificación de emociones a partir de respuestas textuales proporcionadas por estudiantes. Se implementan tanto algoritmos tradicionales con TF-IDF como modelos de lenguaje preentrenados (Transformers) para identificar emociones como Felicidad, Tristeza, Ira, Miedo, Disgusto y Sorpresa.

## 🧪 Tecnologías utilizadas

- Python 3.10
- scikit-learn
- pandas, numpy, matplotlib, seaborn
- Hugging Face Transformers
- NLTK y WordCloud
- Google Colab

## 📂 Estructura del proyecto

- `data/`: Archivos del dataset en bruto y limpio.
- `notebooks/`: Scripts Colab organizados por etapa del proyecto.
- `utils/`: Funciones reutilizables de procesamiento de texto.
- `results/`: Métricas, gráficas y tablas generadas.
- `models/`: Checkpoints de modelos entrenados.
- `README.md`: Esta guía.
- `requirements.txt`: Dependencias Python necesarias.

## 🚀 Cómo ejecutar

1. Clonar el repositorio o abrir el notebook principal en Google Colab.
2. Asegurarse de cargar el dataset limpio en `data/`.
3. Ejecutar cada notebook según el orden numerado (01 → 05).
4. Visualizar resultados en la carpeta `results/`.

## 📈 Modelos comparados

- **Algoritmos tradicionales:** SVM, KNN, Naive Bayes, Random Forest (con TF-IDF).
- **Modelos de Transformers:** BETO, RoBERTuito, MarIA.
- **Embeddings:** Representaciones generadas por modelos para alimentar clasificadores tradicionales.

## 🧠 Resultados esperados

Se espera que los modelos basados en Transformers superen a los tradicionales en métricas como Accuracy y F1-macro, especialmente sin preprocesamiento agresivo. Además, se evalúa una tercera estrategia combinando embeddings con modelos clásicos.


