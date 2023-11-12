import tensorflow as tf

def convolutional_autoencoder(input_shape):
    model = tf.keras.Sequential()

    # Encoding Stage
    model.add(tf.keras.layers.Conv2D(30, kernel_size=(1, 30), strides=(1, 5), input_shape=input_shape, padding='same'))
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
