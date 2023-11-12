import tensorflow as tf

def create_model(input_shape, t1, f1, t2, f2, nn_units):
    model = tf.keras.Sequential()

    # Capa 1: Vertical Convolution Layer
    model.add(tf.keras.layers.Conv2D(filters=f1, kernel_size=(t1, 1), strides=(1, 1), input_shape=input_shape, padding='valid'))

    # Capa 2: Horizontal Convolution Layer
    model.add(tf.keras.layers.Conv2D(filters=f2, kernel_size=(1, t2), strides=(1, 1), padding='valid'))

    # Capa 3: First Fully Connected Layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=nn_units, activation='relu'))

    # Capa 4: Second Fully Connected Layer
    model.add(tf.keras.layers.Dense(units=model.layers[-2].output_shape[1] * model.layers[-2].output_shape[2] * model.layers[-2].output_shape[3], activation='relu'))
    model.add(tf.keras.layers.Reshape(target_shape=model.layers[-3].output_shape[1:]))

    # Capa 5: Deconvolution corresponding to Horizontal Convolution Layer
    model.add(tf.keras.layers.Conv2DTranspose(filters=f2, kernel_size=(1, t2), strides=(1, 1), padding='valid'))

    # Capa 6: Deconvolution corresponding to Vertical Convolution Layer
    model.add(tf.keras.layers.Conv2DTranspose(filters=f1, kernel_size=(t1, 1), strides=(1, 1), padding='valid'))

    return model

# Definir las dimensiones de entrada y otros parámetros según el paper
input_shape = (F, T, 1)  # Ajustar F y T según el contexto del otro paper
t1, f1 = 30, 64  # Ajustar según el contexto del otro paper
t2, f2 = 20, 32  # Ajustar según el contexto del otro paper
nn_units = 128  # Ajustar según el contexto del otro paper

# Crear el modelo
model = create_model(input_shape, t1, f1, t2, f2, nn_units)

# Imprimir resumen del modelo
model.summary()
