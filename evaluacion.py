import torchaudio
from torchmetrics.audio import SignalDistortionRatio
from torchmetrics.audio import SignalNoiseRatio
import librosa
import matplotlib.pyplot as plt

def calcular_metricas(path_pred,path_original):

    # Levanto los audios:
    original, _ = torchaudio.load(path_original)
    pred, _ = torchaudio.load(path_pred)

    # Ajusto longitudes para que sean iguales para poder calcular métricas
    min_length = min(original.size(1), pred.size(1))
    original = original[:, :min_length]
    pred = pred[:, :min_length]

    # Obtengo métricas:
    snr = SignalNoiseRatio()
    sdr = SignalDistortionRatio()
    print('SNR:',snr(pred,original))
    print('SDR:',sdr(pred,original))


def obtener_espectrogramas(path_pred,path_original,instrumento,combinacion):

    # Levanto los audios:
    pred,_ = librosa.load(path_pred,duration=30,sr=22050,mono=True)
    original,_ = librosa.load(path_original,duration=30,sr=22050,mono=True)


    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    fig.patch.set_facecolor('white')

    _,_,_,_ = ax[0].specgram(original, NFFT=1024, Fs=22050, noverlap=256)
    ax[0].set_title(f'Espectograma de {instrumento} original en mix {combinacion}')
    ax[0].set_xlabel('Tiempo [s]')
    ax[0].set_ylabel('Amplitud [Hz]')
    ax[0].grid(False)

    _,_,_,_ = ax[1].specgram(pred, NFFT=1024, Fs=22050, noverlap=256)
    ax[1].set_title(f'Espectrograma de {instrumento} separado en mix {combinacion}')
    ax[1].set_xlabel('Tiempo [s]')
    ax[1].set_ylabel('Amplitud [Hz]')
    ax[1].grid(False)

    plt.tight_layout()
    plt.show();