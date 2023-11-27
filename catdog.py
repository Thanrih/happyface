import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy

#Definindo diretório raiz
dataset_dir = os.path.join(os.getcwd(), 'DataSet')
dataset_treino_dir = os.path.join(dataset_dir, 'treino')

#estou pegando o caminho do dataset de treino, tanto de gatos quanto de cães, e estou contando o número de elementos contidos nele. 

dataset_treino_cats_len = len(os.listdir(os.path.join(dataset_treino_dir, 'cat')))
dataset_treino_dogs_len = len(os.listdir(os.path.join(dataset_treino_dir, 'dog')))

dataset_validacao_dir = os.path.join(dataset_dir, 'validacao')
dataset_validacao_cats_len = len(os.listdir(os.path.join(dataset_validacao_dir, 'cat')))
dataset_validacao_dogs_len = len(os.listdir(os.path.join(dataset_validacao_dir, 'dog')))


print(f'Train Cat: {dataset_treino_cats_len}')
print(f'Train Dogs:{dataset_treino_dogs_len}')
print(f'validação Cat:{dataset_validacao_cats_len}')
print(f'validação Cat:{dataset_validacao_dogs_len}')


#preparando o dataset deixando tudo em 160x160
image_width = 160
image_height = 160
image_color_channel = 3
image_color_channel_size = 255
image_size = (image_width, image_height)
image_shape = image_size + (image_color_channel,)

batch_size = 32 #quantidade de imagens que virão por vez do dataset
epochs = 20 #número de vezes que passarei pelo dataset (epocas])
learning_rate = 0.0001

class_names = ['cat','dog'] #quero que a saída seja uma string, e não um valor numérico


#usado para treinar
dataset_treino =tf.keras.preprocessing.image_dataset_from_directory(
    dataset_treino_dir,
    image_size = image_size,
    batch_size = batch_size,
    shuffle = True
)

#verifica se a imagem não está corrompida (obrigado Deus)
def is_valid_image(img_path):
    try:
        img = Image.open(img_path)
        img.verify()
        return True
    except (IOError, SyntaxError) as e:
        print(f"Invalid image: {img_path}")
        return False

# Verificar e carregar apenas imagens válidas
valid_image_paths = [img_path for img_path in dataset_treino.file_paths if is_valid_image(img_path)]
print('--------------------')

#usado para validar o treinamento
dataset_validacao =tf.keras.preprocessing.image_dataset_from_directory(
    dataset_validacao_dir,
    image_size = image_size,
    batch_size = batch_size,
    shuffle = True
)
dataset_validacao_cardinality = tf.data.experimental.cardinality(dataset_validacao)
dataset_validacao_batches = dataset_validacao_cardinality // 5

dataset_teste = dataset_validacao.take(dataset_validacao_batches)
dataset_validacao = dataset_validacao.skip(dataset_validacao_batches)

print(f'Validacao Dataset cardinality {tf.data.experimental.cardinality(dataset_validacao)}')
print(f'Test Dataset Cardinaliti {tf.data.experimental.cardinality(dataset_teste)}')

def plot_datasets(dataset):
    plt.gcf().clear()
    plt.figure(figsize=(15,15))

    for features, labels in dataset.take(1):
        for i in range(9):
            plt.subplot(3, 3, i +1)
            plt.axis('off')

            plt.imshow(features[i].numpy().astype('uint8'))
            plt.title(class_names[labels[i]])
    plt.show()

#plot_datasets(dataset_treino)
#plot_datasets(dataset_validacao)
#plot_datasets(dataset_teste)

#cada camada será aplicada uma após a outra
model = tf.keras.models.Sequential([
    
    #normalizando os valores de tamnho e de cores das imagens (não quero que vá de 0 a 255, quero de 0 a 1)
    tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / image_color_channel_size, input_shape=image_shape),

    #passando a kernel 16 vezes, com tamanho 3 de forma convolucional, de ativação re, cuja função é a de zerar valores negativos na imagem,
    #permanecendo entre 0 e 1
    tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
    #aplica o activation map da camada de forma otimizada de forma a evitar o overfitting (atentar a detalhes inuteis)
    tf.keras.layers.MaxPooling2D(),
    #aplico mais camadas
    tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    #   converte dados multidimensionais em vetores unidimensionais.
    #  Ela achata os dados :D. Antes de alimentar od dados em camadas densmentente conectadas, é bom fazer isso.
    #  Se não dá pau.
    tf.keras.layers.Flatten(),
    #  Camada com 128 nós, usado para armazenar e atualizar o resultado dos dados processados
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    #  Camada com 1 nó, pois eu só quero uma saída. Uso a função sigmoide para receber valores entre -1 e 1 (usei relu antes então vai ter valores entre 0 e 1, apenas)
    tf.keras.layers.Dense(1, activation='sigmoid')
])    
model.compile(
    #otimiza os erros. Adam procura as melhores configurações para otimizar o calculo.
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
    #serve para calculo de custo otimizada para saída binária (0 ou 1). 
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)
model.summary()

history = model.fit(
    dataset_treino,
    validation_data=dataset_validacao,
    epochs=epochs
)

def plot_dataset_preditions(dataset):
    features, labels = dataset.as_numpy_iterator().next()

    predicao = model.predict_on_batch(features).flatten()
    predicao = tf.where(predicao < 0.5, 0, 1)

    print(f'Labels: {labels}')
    print(f'Predicoes: {predicao.numpy()}')

    plt.figure(figsize=(15,15))

    for i in range (9):
        plt.subplot(3, 3, i + 1)
        plt.axis('off')

        plt.imshow(features[i].astype('uint8'))
        plt.title(class_names[predicao[i]])
    plt.show()

plot_dataset_preditions(dataset_teste)
model.save('path/to/model')
model=tf.keras.models.load_model('path/to/model')