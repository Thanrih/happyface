import os
import matplotlib.pyplot as plt
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

image_width, image_height = 160, 160
image_color_channel = 3
batch_size = 32
epochs = 20
learning_rate = 0.0003
class_names = ['happy', 'sad']
model_path = 'path/to/model'

# Os diretórios
dataset_dir = os.path.join(os.getcwd(), 'DataSet')
dataset_treino_dir = os.path.join(dataset_dir, 'treino')
dataset_validacao_dir = os.path.join(dataset_dir, 'validacao')

# Carregar o dataset
def load_dataset(directory):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=(image_width, image_height),
        batch_size=batch_size,
        shuffle=True
    )

# puxando os diretórios de treino e validação
dataset_treino = load_dataset(dataset_treino_dir)
dataset_validacao = load_dataset(dataset_validacao_dir)

# função pra mostrar os datasets que eu quiser
def plot_datasets(dataset):
    plt.figure(figsize=(15, 15))
    for features, labels in dataset.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.axis('off')
            plt.imshow(features[i].numpy().astype('uint8'))
            plt.title(class_names[labels[i]])
    plt.show()

# Plot datasets()
# plot_datasets(dataset_treino)
# plot_datasets(dataset_validacao)

# estou aprimorando a análise das imagens dando zoom e as rotacionando
data_augmentation = tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2)
])

# agora realmente começa o código, e estou verificando se ele já tem algum save, se não, vai pro treinamento
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Modelo carregado com sucesso!")
    model.summary()
else:
    print("O modelo ainda não foi treinado.")

    # Estou usando um método de transferencia de aprendizado, usando dados de uma rede neural já treinada com milhares de outras imagens
    base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(image_width, image_height, image_color_channel),
    include_top=False,
    weights='imagenet'
    )
    base_model.trainable = False

    # estou processando as imagens, convertendo elas para um formato entendível e processável
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / (image_color_channel / 2), offset=-1, input_shape=(image_width, image_height, image_color_channel)),
        data_augmentation,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    #otimizando a bomba de dados que é a model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # histórico pra treinamento
    history = model.fit(
        dataset_treino,
        validation_data=dataset_validacao,
        epochs=epochs
    )

    # exibição de histórico
    def plot_model():
        plt.figure(figsize=(15, 8))

        plt.subplot(1, 2, 1)
        plt.title('Precisão de treino e validação')
        plt.plot(history.history['accuracy'], label='precisão de treino')
        plt.plot(history.history['val_accuracy'], label='precisão de validação')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        plt.title('Valor de perda no treino e validação')
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Val loss')
        plt.legend(loc='lower right')
        plt.show()

    plot_model()
    model.save(model_path)

# Função pra mostrar o dataset no final de tudo
def plot_dataset_predictions_test(dataset):
    features, labels = dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(features).flatten()
    predictions = tf.where(predictions < 0.5, 0, 1)

    print(f'Labels: {labels}')
    print(f'Predictions: {predictions.numpy()}')

    plt.figure(figsize=(15, 15))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.axis('off')
        plt.imshow(features[i].astype('uint8'))
        plt.title(class_names[predictions[i]])
    plt.show()

# mostrar o resultado final
plot_dataset_predictions_test(dataset_validacao.take(tf.data.experimental.cardinality(dataset_validacao) // 9))
