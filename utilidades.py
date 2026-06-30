import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tabulate import tabulate

# Devolvemos a função que o main.py estava sentindo falta:
def configurar_ambiente():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def exibir_resultados(resultados):
    headers = ["Modelo/Config", "Acurácia", "Precisão", "Recall", "Treino(s)", "Inferência(s)"]
    print("\n" + tabulate(resultados, headers=headers, floatfmt=".6f"))
    print()

def salvar_matriz_confusao(y_true, y_pred, modelo_nome, base_nome):
    pasta_destino = os.path.join("resultados_graficos", base_nome)
    os.makedirs(pasta_destino, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if base_nome == "Esportes":
        labels = ['Vôlei', 'Basquete', 'Cavalos', 'Judô', 'Fórmula 1', 'Golfe', 'Surfe']
    elif base_nome == "Diabetes":
        labels = ['Negativo', 'Positivo']
    elif base_nome == "Spam":
        labels = ['Legítimo', 'Spam']
    else:
        labels = "auto"

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Matriz de Confusão - {modelo_nome} ({base_nome})')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    caminho_arquivo = os.path.join(pasta_destino, f"{modelo_nome}_{base_nome}.png")
    plt.savefig(caminho_arquivo, dpi=150)
    plt.close()

def plotar_comparativo_historico(historicos, base_nome):
    pasta_destino = os.path.join("resultados_graficos", base_nome)
    os.makedirs(pasta_destino, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for nome, hist in historicos.items():
        plt.plot(hist.history['val_accuracy'], label=nome)
    plt.title(f'Acurácia de Validação - {base_nome}')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    for nome, hist in historicos.items():
        plt.plot(hist.history['val_loss'], label=nome)
    plt.title(f'Loss de Validação - {base_nome}')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    caminho_arquivo = os.path.join(pasta_destino, f"historico_keras_{base_nome}.png")
    plt.savefig(caminho_arquivo, dpi=150)
    plt.close()