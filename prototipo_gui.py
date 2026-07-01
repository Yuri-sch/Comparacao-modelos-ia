import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class AppClassificadorEsportes:
    def __init__(self, root):
        self.root = root
        self.root.title("Classificador de Esportes - Modelo CNN")
        self.root.geometry("500x600")
        self.root.configure(bg="#f0f2f5")
        self.classes = ['Vôlei', 'Basquete', 'Corrida de Cavalos', 'Judô', 'Fórmula 1', 'Golfe', 'Surfe']
        self.modelo = None
        self.caminho_imagem = None

        self.construir_interface()
        self.carregar_modelo()

    def carregar_modelo(self):
        try:
            self.modelo = load_model("modelo_esportes.keras")
            self.lbl_status.config(text="Modelo Carregado com Sucesso!", fg="green")
        except Exception as e:
            self.lbl_status.config(text="Erro: Arquivo 'modelo_esportes.keras' não encontrado.", fg="red")
            messagebox.showerror("Erro", "Você precisa rodar a Opção 6 do menu principal para gerar o modelo antes de usar o protótipo!")

    def construir_interface(self):
        # Título
        tk.Label(self.root, text="Classificador de Esportes", font=("Arial", 16, "bold"), bg="#f0f2f5").pack(pady=20)
        
        # Status
        self.lbl_status = tk.Label(self.root, text="Carregando Modelo...", font=("Arial", 10, "italic"), bg="#f0f2f5")
        self.lbl_status.pack(pady=5)

        # Frame da Imagem
        self.frame_img = tk.Frame(self.root, width=300, height=300, bg="white", relief="groove", bd=2)
        self.frame_img.pack(pady=10)
        self.frame_img.pack_propagate(False) # Mantém o tamanho fixo
        
        self.lbl_imagem = tk.Label(self.frame_img, text="Nenhuma imagem carregada", bg="white")
        self.lbl_imagem.pack(expand=True, fill="both")

        # Botão Carregar
        btn_carregar = tk.Button(self.root, text="Selecionar Imagem", font=("Arial", 12), command=self.selecionar_imagem, bg="#007bff", fg="white", cursor="hand2")
        btn_carregar.pack(pady=10)

        # Resultado
        self.lbl_resultado = tk.Label(self.root, text="-", font=("Arial", 18, "bold"), fg="#333", bg="#f0f2f5")
        self.lbl_resultado.pack(pady=20)

    def selecionar_imagem(self):
        self.caminho_imagem = filedialog.askopenfilename(
            title="Escolha uma foto de esporte",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png")]
        )
        
        if self.caminho_imagem:
            self.exibir_imagem()
            self.fazer_predicao()

    def exibir_imagem(self):
        img = Image.open(self.caminho_imagem)
        img = img.resize((300, 300), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.lbl_imagem.config(image=img_tk, text="")
        self.lbl_imagem.image = img_tk # Mantém a referência

    def fazer_predicao(self):
        if self.modelo is None:
            return

        try:
            img = Image.open(self.caminho_imagem).resize((64, 64))
            img_array = img_to_array(img) / 255.0
            img_tensor = np.expand_dims(img_array, axis=0)
            predicao_probs = self.modelo.predict(img_tensor, verbose=0)
            indice_classe = np.argmax(predicao_probs[0])
            certeza = predicao_probs[0][indice_classe] * 100

            esporte_previsto = self.classes[indice_classe]
            self.lbl_resultado.config(text=f"Classificação: {esporte_previsto}\n(Confiança: {certeza:.1f}%)")

        except Exception as e:
            messagebox.showerror("Erro de Processamento", f"Não foi possível classificar a imagem.\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AppClassificadorEsportes(root)
    root.mainloop()