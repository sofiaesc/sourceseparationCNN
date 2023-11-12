import tensorflow as tf

def convolutional_autoencoder(input_shape):
    model = tf.keras.Sequential()

    # Encoding Stage
    model.add(tf.keras.layers.Conv2D(
    30,                   # Número de filtros (neuronas) en la capa. En este caso, se utilizan 30 filtros.
    kernel_size=(1, 30),  # Tamaño del kernel de convolución. En este caso, es una convolución 1x30.
    strides=(1, 5),       # Longitud del paso durante la convolución. (1, 5) significa un paso de 1 en la dimensión vertical y 5 en la dimensión horizontal.
    input_shape=input_shape,  # Forma de entrada esperada para esta capa. En este caso, se espera que la entrada tenga la forma especificada por la variable input_shape.
    padding='same'        # Método de relleno. 'same' significa que se aplicará relleno para mantener el tamaño de la salida igual que el tamaño de la entrada, si es necesario.
    ))
    model.add(tf.keras.layers.Conv2D(30, kernel_size=(2, 3 * input_shape[1]), strides=(1, 1), padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))

    # Decoding Stage
    model.add(tf.keras.layers.Reshape((1, 1, 256)))
    model.add(tf.keras.layers.Conv2DTranspose(30, kernel_size=(2, 3 * input_shape[1]), strides=(1, 1), padding='valid'))
    model.add(tf.keras.layers.Conv2DTranspose(30, kernel_size=(1, 30), strides=(1, 5), padding='same'))

    return model

# Assuming input spectrogram shape is (F, T, 1), adjust it accordingly
input_shape = (F, T, 1)

# Create the model
model = convolutional_autoencoder(input_shape)

# Print model summary
model.summary()

# Compile the model with an appropriate optimizer and loss function
optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
model.compile(optimizer=optimizer, loss='mean_squared_error')
