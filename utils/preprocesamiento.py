
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

def limpiar_texto(texto):
    """
    Limpia un texto dado eliminando caracteres especiales, números,
    múltiple espacio y convierte a minúsculas.

    Parámetros:
        texto (str): Texto original a procesar.

    Retorna:
        str: Texto limpio.
    """
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)  # eliminar números
    texto = re.sub(rf"[{re.escape(string.punctuation)}]", '', texto)  # eliminar puntuación
    texto = re.sub(r'\s+', ' ', texto).strip()  # espacios múltiples
    return texto

def tokenizar_y_lemmatizar(texto, stopwords_es):
    """
    Tokeniza y lematiza un texto aplicando eliminación de stopwords.

    Parámetros:
        texto (str): Texto limpio.
        stopwords_es (set): Conjunto de palabras vacías en español.

    Retorna:
        str: Texto transformado.
    """
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(texto)
    tokens_filtrados = [lemmatizer.lemmatize(t) for t in tokens if t not in stopwords_es]
    return ' '.join(tokens_filtrados)

def generar_tfidf_features(textos):
    """
    Convierte una lista de textos en una matriz TF-IDF.

    Parámetros:
        textos (list): Lista de textos preprocesados.

    Retorna:
        sparse matrix: Representación TF-IDF de los textos.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(textos)
    return X

def entrenar_modelo_clasificacion(modelo, X_train, y_train, X_test, y_test):
    """
    Entrena y evalúa un modelo de clasificación.

    Parámetros:
        modelo (sklearn): Modelo instanciado.
        X_train (array): Features de entrenamiento.
        y_train (array): Etiquetas de entrenamiento.
        X_test (array): Features de prueba.
        y_test (array): Etiquetas de prueba.

    Retorna:
        dict: Métricas de rendimiento (accuracy y F1).
    """
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    return {"accuracy": acc, "f1_macro": f1}
