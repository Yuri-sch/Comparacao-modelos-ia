# main.py (Atualizado)
import os
import sys

# Importações dos módulos que criamos
from pre_processamento import carregar_diabetes, carregar_spam, carregar_esportes
from modelos import (treinar_rf_diabetes, treinar_mlp_diabetes, treinar_keras_diabetes,
                     treinar_rf_spam, treinar_mlp_spam, treinar_keras_spam,
                     treinar_rf_esportes, treinar_mlp_esportes, treinar_keras_esportes)
from utilidades import configurar_ambiente

def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

def menu_principal():
    while True:
        limpar_tela()
        print("="*55)
        print("   SISTEMA DE TREINAMENTO - TRABALHO DE ML")
        print("="*55)
        print("\n[ BASE 1: DIABETES (Tabular) ]")
        print("  1. Treinar Random Forest (3 configs)")
        print("  2. Treinar Sklearn MLPClassifier (2 arquiteturas)")
        print("  3. Treinar Keras Redes Neurais (3 Arquiteturas)")
        
        print("\n[ BASE 2: ESPORTES (Imagens) ]")
        print("  4. Treinar Random Forest (Flatten)")
        print("  5. Treinar Sklearn MLPClassifier (2 arquiteturas)")
        print("  6. Treinar Keras CNN (3 Arquiteturas Convolucionais)")
        
        print("\n[ BASE 3: SPAM (Texto) ]")
        print("  7. Treinar Random Forest (3 configs)")
        print("  8. Treinar Sklearn MLPClassifier (2 arquiteturas)")
        print("  9. Treinar Keras Redes Neurais (3 Arquiteturas)")
        
        print("\n  0. Sair")
        print("="*55)
        
        opcao = input("\nEscolha qual modelo executar agora: ")
        
        # --- BASE 1 ---
        if opcao in ['1', '2', '3']:
            X_train, X_test, y_train, y_test = carregar_diabetes("diabetes_prediction_dataset.csv")
            if opcao == '1': treinar_rf_diabetes(X_train, X_test, y_train, y_test)
            if opcao == '2': treinar_mlp_diabetes(X_train, X_test, y_train, y_test)
            if opcao == '3': treinar_keras_diabetes(X_train, X_test, y_train, y_test)
            input("\nPressione ENTER para voltar ao menu...")
            
        # --- BASE 2 ---
        elif opcao in ['4', '5', '6']:
            caminho_imagens = "sports_data/train"
            X_train, X_test, y_train, y_test = carregar_esportes(caminho_imagens)
            
            if len(X_train) > 0:
                if opcao == '4': treinar_rf_esportes(X_train, X_test, y_train, y_test)
                if opcao == '5': treinar_mlp_esportes(X_train, X_test, y_train, y_test)
                if opcao == '6': treinar_keras_esportes(X_train, X_test, y_train, y_test)
            input("\nPressione ENTER para voltar ao menu...")
            
        # --- BASE 3 ---
        elif opcao in ['7', '8', '9']:
            X_train, X_test, y_train, y_test = carregar_spam("spam_email_dataset.csv")
            if opcao == '7': treinar_rf_spam(X_train, X_test, y_train, y_test)
            if opcao == '8': treinar_mlp_spam(X_train, X_test, y_train, y_test)
            if opcao == '9': treinar_keras_spam(X_train, X_test, y_train, y_test)
            input("\nPressione ENTER para voltar ao menu...")
            
        elif opcao == '0':
            print("\nEncerrando a aplicação.")
            sys.exit()
            
        else:
            print("\nOpção inválida.")
            input("\nPressione ENTER para continuar...")

if __name__ == "__main__":
    configurar_ambiente()
    menu_principal()