import os, sys
import argparse
import numpy as np
import onnxruntime as ort
import librosa
import tqdm
import soundfile as sf
import onnxruntime as ort


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input audio file")
    parser.add_argument("--output", "-o", type=str, required=False, default="clean.wav", help="Output audio file")
    return parser.parse_args()


def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Compute STFT using librosa
    stft_spec = librosa.stft(y, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
                             window='hann', center=center)
    
    # Get magnitude and phase from complex spectrogram
    mag, pha = librosa.magphase(stft_spec)
    
    # Apply small constant to avoid log(0) or division by zero
    mag = np.abs(stft_spec) + 1e-9
    pha = np.angle(stft_spec + 1e-5)  # Add a small constant for numerical stability
    
    # Magnitude Compression
    mag = np.power(mag, compress_factor)

    return mag, pha


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = np.power(mag, (1.0 / compress_factor))
    
    # Combine magnitude and phase into complex numbers
    com = mag * (np.cos(pha) + 1j * np.sin(pha))
    
    # Perform the inverse STFT using librosa
    wav = librosa.istft(com, hop_length=hop_size, win_length=win_size, window='hann', center=center)

    return wav


def main():
    args = get_args()
    assert os.path.exists(args.input), f"Input audio file {args.input} not exist"

    input_audio_file = args.input
    output_audio_file = args.output

    # Load model
    sess = ort.InferenceSession("mp-senet.onnx", providers=["CPUExecutionProvider"])
    slice_len = sess.get_inputs()[0].shape[-1]

    # from config.json
    sampling_rate = 16000
    n_fft = 400
    hop_size = 100
    win_size = 400
    compress_factor = 0.3

    # Load audio and preprocess
    noisy_wav, _ = librosa.load(input_audio_file, sr=sampling_rate)
    norm_factor = np.sqrt(noisy_wav.shape[0] / np.sum(noisy_wav ** 2.0))
    noisy_wav = (noisy_wav * norm_factor)[None, ...]
    noisy_mag, noisy_pha = mag_pha_stft(noisy_wav, n_fft, hop_size, win_size, compress_factor)
    # print(f"noisy_mag.shape = {noisy_mag.shape}")
    # print(f"noisy_pha.shape = {noisy_pha.shape}")
    total_input_len = noisy_mag.shape[-1]

    slice_num = int(np.ceil(total_input_len / slice_len))
    amp_list = []
    pha_list = []
    if total_input_len < slice_num * slice_len:
        pad_size = ((0,0), (0,0), (0, slice_num * slice_len - total_input_len))
        noisy_mag = np.pad(noisy_mag, pad_size, mode="constant", constant_values=0)
        noisy_pha = np.pad(noisy_pha, pad_size, mode="constant", constant_values=0)

    for i in tqdm.trange(slice_num):
        sub_mag = noisy_mag[..., i * slice_len : (i + 1) * slice_len]
        sub_pha = noisy_pha[..., i * slice_len : (i + 1) * slice_len]

        amp_g, pha_g_i, pha_g_r = sess.run(None, {"noise_mag": sub_mag, "noise_pha": sub_pha})
        amp_list.append(amp_g)
        pha_list.append(np.arctan2(pha_g_i, pha_g_r))

    amp_g = np.concatenate(amp_list, axis=-1)[:total_input_len]
    pha_g = np.concatenate(pha_list, axis=-1)[:total_input_len]
    audio_g = mag_pha_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
    audio_g = audio_g / norm_factor
    audio_g = audio_g[0, :noisy_wav.shape[-1]]

    # print(f"audio_g.shape = {audio_g.shape}")

    sf.write(output_audio_file, audio_g, samplerate=sampling_rate, format='PCM_16')
    print(f"Save output audio to {output_audio_file}")


if __name__ == "__main__":
    main()