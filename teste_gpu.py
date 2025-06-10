import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Dispositivos:", tf.config.list_physical_devices())
print("GPUs:", tf.config.list_physical_devices('GPU'))
