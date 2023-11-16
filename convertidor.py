from moviepy.editor import VideoFileClip

def extraer_audio(nombre_archivo_mp4, nombre_salida_wav):
    video_clip = VideoFileClip(nombre_archivo_mp4)
    audio_clip = video_clip.audio

    # Extraer las primeras 500 muestras (0 a 500)
    audio_clip = audio_clip.subclip(0, 3)  # 0.02 segundos si la frecuencia de muestreo es 44100 Hz

    # Guardar el audio en un archivo WAV
    audio_clip.write_audiofile(nombre_salida_wav, codec='pcm_s16le')

    # Cerrar los clips para liberar recursos
    audio_clip.close()
    video_clip.close()

# Especifica el nombre del archivo mp4 de entrada y el nombre del archivo de salida wav
nombre_archivo_mp4 = "song1.mp4"
nombre_salida_wav = "song1_corta.wav"

# Llama a la función para extraer el audio
extraer_audio(nombre_archivo_mp4, nombre_salida_wav)
