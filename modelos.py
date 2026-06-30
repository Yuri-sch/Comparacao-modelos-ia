import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from utilidades import salvar_matriz_confusao, plotar_comparativo_historico, exibir_resultados

# =====================================================================
# BASE 1: DIABETES (Tabular)
# =====================================================================

def treinar_rf_diabetes(X_train, X_test, y_train, y_test):
    print("Treinando Random Forest (Diabetes) - 3 Configurações...")
    resultados = []
    configs = {
        "RF_config_1": RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, random_state=42),
        "RF_config_2": RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, random_state=42),
        "RF_config_3": RandomForestClassifier(n_estimators=200, criterion='gini', min_samples_split=5, random_state=42)
    }
    for nome, model in configs.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        t_treino = time.time() - t0
        
        t1 = time.time()
        preds = model.predict(X_test)
        t_inf = time.time() - t1
        
        resultados.append([nome, accuracy_score(y_test, preds), precision_score(y_test, preds, zero_division=0), recall_score(y_test, preds, zero_division=0), t_treino, t_inf])
        salvar_matriz_confusao(y_test, preds, nome, "Diabetes")
    exibir_resultados(resultados)

def treinar_mlp_diabetes(X_train, X_test, y_train, y_test):
    print("Treinando MLP Scikit-Learn (Diabetes) - 2 Arquiteturas...")
    resultados = []
    configs = {
        "MLP_Sklearn_Arq1": MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=300, random_state=42),
        "MLP_Sklearn_Arq2": MLPClassifier(hidden_layer_sizes=(100, 50), activation='tanh', alpha=0.001, max_iter=300, random_state=42)
    }
    for nome, model in configs.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        t_treino = time.time() - t0
        
        t1 = time.time()
        preds = model.predict(X_test)
        t_inf = time.time() - t1
        
        resultados.append([nome, accuracy_score(y_test, preds), precision_score(y_test, preds, zero_division=0), recall_score(y_test, preds, zero_division=0), t_treino, t_inf])
        salvar_matriz_confusao(y_test, preds, nome, "Diabetes")
    exibir_resultados(resultados)

def treinar_keras_diabetes(X_train, X_test, y_train, y_test):
    print("Treinando Keras Redes Neurais (Diabetes) - 3 Arquiteturas...")
    resultados, historicos = [], {}
    input_dim = X_train.shape[1]
    
    arqs = {
        "Keras_Arq_1": [Dense(16, activation='relu', input_dim=input_dim), Dense(1, activation='sigmoid')],
        "Keras_Arq_2": [Dense(64, activation='relu', input_dim=input_dim), Dropout(0.3), Dense(32, activation='relu'), Dense(1, activation='sigmoid')],
        "Keras_Arq_3": [Dense(128, activation='relu', input_dim=input_dim), Dense(64, activation='relu'), Dense(32, activation='relu'), Dense(1, activation='sigmoid')]
    }
    for nome, camadas in arqs.items():
        model = Sequential(camadas)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        t0 = time.time()
        hist = model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=32, verbose=0)
        historicos[nome] = hist
        t_treino = time.time() - t0
        
        t1 = time.time()
        preds = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
        t_inf = time.time() - t1
        
        resultados.append([nome, accuracy_score(y_test, preds), precision_score(y_test, preds, zero_division=0), recall_score(y_test, preds, zero_division=0), t_treino, t_inf])
        salvar_matriz_confusao(y_test, preds, nome, "Diabetes")
        
    plotar_comparativo_historico(historicos, "Diabetes")
    exibir_resultados(resultados)


# =====================================================================
# BASE 2: ESPORTES (Imagens)
# =====================================================================

def treinar_rf_esportes(X_train, X_test, y_train, y_test):
    print("Treinando Random Forest (Esportes) - 3 Configurações...")
    X_tr_flat = X_train.reshape(X_train.shape[0], -1)
    X_te_flat = X_test.reshape(X_test.shape[0], -1)
    
    resultados = []
    configs = {
        "RF_config_1": RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, random_state=42, n_jobs=-1),
        "RF_config_2": RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=15, random_state=42, n_jobs=-1),
        "RF_config_3": RandomForestClassifier(n_estimators=150, criterion='gini', min_samples_split=4, random_state=42, n_jobs=-1)
    }
    for nome, model in configs.items():
        t0 = time.time()
        model.fit(X_tr_flat, y_train)
        t_treino = time.time() - t0
        
        t1 = time.time()
        preds = model.predict(X_te_flat)
        t_inf = time.time() - t1
        
        resultados.append([nome, accuracy_score(y_test, preds), precision_score(y_test, preds, average='weighted', zero_division=0), recall_score(y_test, preds, average='weighted', zero_division=0), t_treino, t_inf])
        salvar_matriz_confusao(y_test, preds, nome, "Esportes")
    exibir_resultados(resultados)

def treinar_mlp_esportes(X_train, X_test, y_train, y_test):
    print("Treinando MLP Scikit-Learn (Esportes) - 2 Arquiteturas...")
    X_tr_flat = X_train.reshape(X_train.shape[0], -1)
    X_te_flat = X_test.reshape(X_test.shape[0], -1)
    
    resultados = []
    configs = {
        "MLP_Sklearn_Arq1": MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42),
        "MLP_Sklearn_Arq2": MLPClassifier(hidden_layer_sizes=(256, 128), alpha=0.01, max_iter=200, random_state=42)
    }
    for nome, model in configs.items():
        t0 = time.time()
        model.fit(X_tr_flat, y_train)
        t_treino = time.time() - t0
        
        t1 = time.time()
        preds = model.predict(X_te_flat)
        t_inf = time.time() - t1
        
        resultados.append([nome, accuracy_score(y_test, preds), precision_score(y_test, preds, average='weighted', zero_division=0), recall_score(y_test, preds, average='weighted', zero_division=0), t_treino, t_inf])
        salvar_matriz_confusao(y_test, preds, nome, "Esportes")
    exibir_resultados(resultados)

def treinar_keras_esportes(X_train, X_test, y_train, y_test):
    print("Treinando Keras CNN (Esportes) - 3 Arquiteturas...")
    resultados, historicos = [], {}
    input_shape = (64, 64, 3)
    
    arqs = {
        "CNN_Arq_1": [Conv2D(32, (3,3), activation='relu', input_shape=input_shape), MaxPooling2D(2,2), Flatten(), Dense(7, activation='softmax')],
        "CNN_Arq_2": [Conv2D(64, (3,3), activation='relu', input_shape=input_shape), MaxPooling2D(2,2), Conv2D(64, (3,3), activation='relu'), MaxPooling2D(2,2), Flatten(), Dense(32, activation='relu'), Dense(7, activation='softmax')],
        "CNN_Arq_3": [Conv2D(32, (3,3), activation='relu', input_shape=input_shape), MaxPooling2D(2,2), Dropout(0.25), Flatten(), Dense(64, activation='relu'), Dropout(0.5), Dense(7, activation='softmax')]
    }
    for nome, camadas in arqs.items():
        model = Sequential(camadas)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        t0 = time.time()
        hist = model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=32, verbose=0)
        historicos[nome] = hist
        t_treino = time.time() - t0
        
        t1 = time.time()
        preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
        t_inf = time.time() - t1
        
        resultados.append([nome, accuracy_score(y_test, preds), precision_score(y_test, preds, average='weighted', zero_division=0), recall_score(y_test, preds, average='weighted', zero_division=0), t_treino, t_inf])
        salvar_matriz_confusao(y_test, preds, nome, "Esportes")
        
        if nome == "CNN_Arq_2":
            model.save("modelo_esportes.keras")
            
    plotar_comparativo_historico(historicos, "Esportes")
    exibir_resultados(resultados)


# =====================================================================
# BASE 3: SPAM (Texto)
# =====================================================================

def treinar_rf_spam(X_train, X_test, y_train, y_test):
    print("Treinando Random Forest (Spam) - 3 Configurações...")
    resultados = []
    configs = {
        "RF_config_1": RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42, n_jobs=-1),
        "RF_config_2": RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=20, random_state=42, n_jobs=-1),
        "RF_config_3": RandomForestClassifier(n_estimators=200, criterion='gini', min_samples_split=10, random_state=42, n_jobs=-1)
    }
    for nome, model in configs.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        t_treino = time.time() - t0
        
        t1 = time.time()
        preds = model.predict(X_test)
        t_inf = time.time() - t1
        
        resultados.append([nome, accuracy_score(y_test, preds), precision_score(y_test, preds, zero_division=0), recall_score(y_test, preds, zero_division=0), t_treino, t_inf])
        salvar_matriz_confusao(y_test, preds, nome, "Spam")
    exibir_resultados(resultados)

def treinar_mlp_spam(X_train, X_test, y_train, y_test):
    print("Treinando MLP Scikit-Learn (Spam) - 2 Arquiteturas...")
    resultados = []
    configs = {
        "MLP_Sklearn_Arq1": MLPClassifier(hidden_layer_sizes=(64,), max_iter=150, random_state=42),
        "MLP_Sklearn_Arq2": MLPClassifier(hidden_layer_sizes=(128, 32), early_stopping=True, max_iter=200, random_state=42)
    }
    for nome, model in configs.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        t_treino = time.time() - t0
        
        t1 = time.time()
        preds = model.predict(X_test)
        t_inf = time.time() - t1
        
        resultados.append([nome, accuracy_score(y_test, preds), precision_score(y_test, preds, zero_division=0), recall_score(y_test, preds, zero_division=0), t_treino, t_inf])
        salvar_matriz_confusao(y_test, preds, nome, "Spam")
    exibir_resultados(resultados)

def treinar_keras_spam(X_train, X_test, y_train, y_test):
    print("Treinando Keras Redes Neurais (Spam) - 3 Arquiteturas...")
    resultados, historicos = [], {}
    input_dim = X_train.shape[1]
    
    arqs = {
        "Keras_Arq_1": [Dense(16, activation='relu', input_dim=input_dim), Dense(1, activation='sigmoid')],
        "Keras_Arq_2": [Dense(64, activation='relu', input_dim=input_dim), Dropout(0.5), Dense(1, activation='sigmoid')],
        "Keras_Arq_3": [Dense(128, activation='relu', input_dim=input_dim), Dropout(0.3), Dense(32, activation='relu'), Dense(1, activation='sigmoid')]
    }
    for nome, camadas in arqs.items():
        model = Sequential(camadas)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        t0 = time.time()
        hist = model.fit(X_train, y_train, validation_split=0.1, epochs=5, batch_size=32, verbose=0)
        historicos[nome] = hist
        t_treino = time.time() - t0
        
        t1 = time.time()
        preds = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
        t_inf = time.time() - t1
        
        resultados.append([nome, accuracy_score(y_test, preds), precision_score(y_test, preds, zero_division=0), recall_score(y_test, preds, zero_division=0), t_treino, t_inf])
        salvar_matriz_confusao(y_test, preds, nome, "Spam")
        
    plotar_comparativo_historico(historicos, "Spam")
    exibir_resultados(resultados)