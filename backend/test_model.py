# test_model.py
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os

# Definir o caminho para o arquivo do modelo
MODEL_PATH = os.path.join('models', 'fashion_mnist_model.h5')

# Carregar o modelo
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Modelo '{MODEL_PATH}' carregado com sucesso para teste.")
except Exception as e:
    print(f"ERRO: Não foi possível carregar o modelo: {e}")
    exit()

# Nomes das classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_image_for_model(image_path):
    try:
        img = Image.open(image_path).convert('L') # Converte para escala de cinza
        img = img.resize((28, 28))             # Redimensiona para 28x28
        img_array = np.array(img)              # Converte para array NumPy
        img_array = img_array / 255.0          # Normaliza pixels para 0-1
        img_array = np.expand_dims(img_array, axis=0)  # Adiciona dimensão do lote
        img_array = np.expand_dims(img_array, axis=-1) # Adiciona dimensão do canal
        return img_array
    except Exception as e:
        print(f"Erro ao pré-processar imagem {image_path}: {e}")
        return None

def test_prediction_with_example_image():
    # Carregue uma imagem de teste Fashion MNIST REAL do seu dataset, se possível.
    # Se não tiver, você pode gerar uma imagem simples de uma bota/tênis programaticamente
    # ou usar uma das imagens de teste do Fashion MNIST se as tiver baixado separadamente.

    # PARA ESTE TESTE, VAMOS REPLICAR UMA IMAGEM DO DATASET ORIGINAL:
    # Baixe uma imagem "oficial" do Fashion MNIST (por exemplo, um tênis ou uma bota)
    # ou use um dos exemplos do dataset de teste que você tem acesso no Colab.
    # Se você não tem uma imagem Fashion MNIST "pura", podemos simular uma.

    # Exemplo: Carregando uma imagem do Fashion MNIST (requer o dataset baixado)
    # Baixe o Fashion MNIST, se ainda não o fez, no seu Colab, e salve uma imagem de teste.
    # Ou, para testar rapidamente, vamos "criar" uma imagem simulada para garantir que o processo está correto.

    # Alternativa mais robusta:
    # Vamos carregar o dataset Fashion MNIST diretamente NO SCRIPT DE TESTE
    # para ter certeza que estamos usando uma imagem com a MESMA DISTRIBUIÇÃO
    # dos dados de treinamento.
    print("\nCarregando dataset Fashion MNIST para obter imagem de teste...")
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalizar as imagens de teste da mesma forma que o treino
    test_images_normalized = test_images / 255.0
    test_images_reshaped = test_images_normalized.reshape((10000, 28, 28, 1))

    # Escolha uma imagem específica do conjunto de teste, por exemplo, um tênis (classe 7)
    # ou uma bota (classe 9) para verificar.
    # Vamos pegar a primeira bota encontrada no conjunto de teste.
    shoe_index = -1
    for i, label in enumerate(test_labels):
        if class_names[label] == 'Ankle boot': # Ou 'Sneaker' para tênis
            shoe_index = i
            break

    if shoe_index != -1:
        print(f"Usando imagem de teste do dataset Fashion MNIST (índice {shoe_index}, classe real: {class_names[test_labels[shoe_index]]})")
        sample_image_for_prediction = test_images_reshaped[shoe_index]
        true_label_for_sample = test_labels[shoe_index]

        # Fazer a previsão (o modelo já espera (1, 28, 28, 1) se pegamos de test_images_reshaped)
        # Se você pegou test_images_normalized[shoe_index] (formato (28,28)), você precisaria fazer:
        # sample_image_for_prediction = np.expand_dims(test_images_normalized[shoe_index], axis=0)
        # sample_image_for_prediction = np.expand_dims(sample_image_for_prediction, axis=-1)

        predictions = model.predict(np.expand_dims(sample_image_for_prediction, axis=0))
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(np.max(predictions[0]))

        print(f"\n--- Resultado do Teste Local ---")
        print(f"Classe Real da Imagem de Teste: {class_names[true_label_for_sample]}")
        print(f"Previsão do Modelo: {predicted_class_name}")
        print(f"Confiança: {confidence:.4f}")
        print(f"Probabilidades: {predictions[0]}")

        if predicted_class_index == true_label_for_sample:
            print("-> O modelo classificou corretamente esta imagem de teste interna!")
        else:
            print("-> ATENÇÃO: O modelo não classificou corretamente esta imagem de teste interna.")
            print(f"   Esperado: {class_names[true_label_for_sample]}, Obtido: {predicted_class_name}")

    else:
        print("Não foi encontrada uma bota no dataset de teste para exemplo.")


if __name__ == '__main__':
    test_prediction_with_example_image()