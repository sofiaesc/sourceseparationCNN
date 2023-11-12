import tensorflow as tf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def convolutional_autoencoder(input_shape):
    model = tf.keras.Sequential()

    # Encoding Stage
    model.add(tf.keras.layers.Conv2D(
        30,                   
        kernel_size=(1, 30),  
        strides=(1, 5),       
        input_shape=input_shape,  
        padding='same'
    ))
    model.add(tf.keras.layers.Conv2D(30, kernel_size=(2, 3 * input_shape[1]), strides=(1, 1), padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))

    # Decoding Stage
    model.add(tf.keras.layers.Reshape((1, 1, 256)))
    model.add(tf.keras.layers.Conv2DTranspose(30, kernel_size=(2, 3 * input_shape[1]), strides=(1, 1), padding='valid'))
    model.add(tf.keras.layers.Conv2DTranspose(30, kernel_size=(1, 30), strides=(1, 5), padding='same'))

    return model


# Cargar un archivo de audio
file_path = 'song1.wav'  # Reemplaza con la ruta de tu archivo de audio
audio, sr = librosa.load(file_path, sr=None, mono=True)

# Calcular la STFT
M = librosa.stft(audio, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='reflect')

# Obtener las dimensiones de la matriz M
print(M.shape)

"""
# Visualizar la STFT
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(M, ref=np.max), y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Espectrograma de la STFT')
plt.show()
"""

# Definir F y T según tus necesidades
F = M.shape[1]
T = M.shape[0]

# Assuming input spectrogram shape is (F, T, 1), adjust it accordingly
input_shape = (F, T, 1)

# Create the model
model = convolutional_autoencoder(input_shape)

# Print model summary
model.summary()

# Compile the model with an appropriate optimizer and loss function
optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Preprocesar el audio
spectrogram = librosa.stft(audio, n_fft=1024, hop_length=512)
magnitude_spectrogram = np.abs(spectrogram)
input_data = np.expand_dims(np.log1p(magnitude_spectrogram), axis=-1)

# Reshape para que coincida con el tamaño de entrada esperado por el modelo
input_data = np.reshape(input_data, (1, input_data.shape[0], input_data.shape[1], 1))

# Hacer predicciones
output_data = model.predict(input_data)

# Postprocesar los datos
output_data = np.squeeze(output_data)
reconstructed_spectrogram = np.expm1(output_data)

# Reconstruir la señal de audio
reconstructed_audio = librosa.istft(reconstructed_spectrogram, hop_length=512)

# Guardar la señal de audio reconstruida
output_path = 'output_audio.wav'
librosa.output.write_wav(output_path, reconstructed_audio, sr)

# Visualizar la entrada y la salida
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(magnitude_spectrogram, ref=np.max), y_axis='log', x_axis='time')
plt.title('Input Spectrogram')
plt.colorbar(format='%+2.0f dB')

plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(reconstructed_spectrogram, ref=np.max), y_axis='log', x_axis='time')
plt.title('Reconstructed Spectrogram')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.show()
