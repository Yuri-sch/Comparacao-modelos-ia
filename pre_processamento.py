import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTEENN

def carregar_diabetes(caminho_csv):
    df = pd.read_csv(caminho_csv)
        
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['smoking_history'] = le.fit_transform(df['smoking_history'])

    X = df.drop(columns=['diabetes'])
    y = df['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Rebalanceamento com SMOTEENN no conjunto de treino
    smote_enn = SMOTEENN(random_state=42)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

    # Padronização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_resampled, y_test

def carregar_spam(caminho_csv):
    df = pd.read_csv(caminho_csv).dropna(subset=['email_text', 'label'])

    X = df['email_text']
    y = df['label'].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Vetorização TF-IDF
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train_raw).toarray()
    X_test_vec = vectorizer.transform(X_test_raw).toarray()

    return X_train_vec, X_test_vec, y_train, y_test

def carregar_esportes(diretorio_base):
    print("\nCarregando e processando imagens... Isso pode levar alguns segundos.")

    # Mapeamento dos seus 7 esportes escolhidos:
    esportes_alvo = [
        'volleyball', 
        'basketball', 
        'horse racing', 
        'judo', 
        'formula 1 racing',
        'golf', 
        'surfing'
    ]

    img_size = (64, 64)
    X, y = [], []

    if not os.path.exists(diretorio_base):
        print(f"ERRO: A pasta '{diretorio_base}' não foi encontrada.")
        return [], [], [], []

    for label_idx, esporte in enumerate(esportes_alvo):
        caminho_pasta = os.path.join(diretorio_base, esporte)
        if not os.path.exists(caminho_pasta):
            print(f"[AVISO] Pasta não encontrada para: {esporte}")
            continue

        arquivos = os.listdir(caminho_pasta)[:200]
        for arquivo in arquivos:
            if arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                caminho_img = os.path.join(caminho_pasta, arquivo)
                try:
                    img = load_img(caminho_img, target_size=img_size)
                    img_array = img_to_array(img) / 255.0
                    X.append(img_array)
                    y.append(label_idx)
                except Exception:
                    pass

    if len(X) == 0:
        print("Nenhuma imagem processada. Verifique os nomes das pastas.")
        return [], [], [], []

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Sucesso! {X_train.shape[0]} imagens de treino e {X_test.shape[0]} de teste carregadas.")
    return X_train, X_test, y_train, y_test