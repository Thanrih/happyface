import os
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

# Verificando disponibilidade da GPU
print("Dispositivos físicos:", tf.config.list_physical_devices())
print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))
# Configurações
image_width, image_height = 160, 160
image_color_channel = 3
batch_size = 32
epochs = 20
learning_rate = 0.0003
class_names = ['happy', 'sad']
model_path = os.path.join(os.getcwd(), 'modelo_salvo')

# Diretórios do dataset
dataset_dir = os.path.join(os.getcwd(), 'DataSet', 'DataSet')
dataset_treino_dir = os.path.join(dataset_dir, 'treino')
dataset_validacao_dir = os.path.join(dataset_dir, 'validacao')

# Carregamento dos datasets
def load_dataset(directory):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=(image_width, image_height),
        batch_size=batch_size,
        shuffle=True
    )

dataset_treino = load_dataset(dataset_treino_dir)
dataset_validacao = load_dataset(dataset_validacao_dir)

# Visualização de amostras do dataset
def plot_datasets(dataset):
    plt.figure(figsize=(15, 15))
    for features, labels in dataset.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.axis('off')
            plt.imshow(features[i].numpy().astype('uint8'))
            plt.title(class_names[labels[i]])
    plt.show()

# Aumento de dados
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])

# Treinamento ou carregamento do modelo
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Modelo carregado com sucesso!")
    model.summary()
else:
    print("O modelo ainda não foi treinado.")

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(image_width, image_height, image_color_channel),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(image_width, image_height, image_color_channel)),
        data_augmentation,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        dataset_treino,
        validation_data=dataset_validacao,
        epochs=epochs
    )

    with open('train_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    def plot_model():
        plt.figure(figsize=(15, 8))

        plt.subplot(1, 2, 1)
        plt.title('Precisão de treino e validação')
        plt.plot(history.history['accuracy'], label='precisão de treino')
        plt.plot(history.history['val_accuracy'], label='precisão de validação')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        plt.title('Perda no treino e validação')
        plt.plot(history.history['loss'], label='perda treino')
        plt.plot(history.history['val_loss'], label='perda validação')
        plt.legend(loc='lower right')
        plt.show()

    plot_model()
    model.save(model_path)

# Predições em lote final
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

plot_dataset_predictions_test(dataset_validacao.take(tf.data.experimental.cardinality(dataset_validacao) // 9))
1