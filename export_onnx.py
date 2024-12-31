import torch
from onnxsim import simplify
import argparse
from models.model import MPNet
import os
import json
import onnx
import librosa
import tarfile as tf
import numpy as np
import tqdm
import soundfile as sf
import onnxruntime as ort


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio", "-i", type=str, default="noisy_snr0.wav")
    parser.add_argument("--output_audio", "-o", type=str, default="output.wav")
    parser.add_argument("--length", "-l", type=int, default=1024)
    parser.add_argument("--checkpoint", "-c", type=str, default="./best_ckpt/g_best_vb")
    parser.add_argument("--model", "-m", type=str, default="mp-senet.onnx")
    return parser.parse_args()


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1]+(1e-10), stft_spec[:, :, :, 0]+(1e-5))
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)

    return mag, pha


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return wav


def main():
    args = get_args()
    input_audio = args.input_audio
    output_audio = args.output_audio
    length = args.length
    ckpt = args.checkpoint
    model_name = args.model
    print(f"input_audio: {input_audio}")
    print(f"output_audio: {output_audio}")
    print(f"length: {length}")
    print(f"ckpt: {ckpt}")
    print(f"model_name: {model_name}")

    device = torch.device('cpu')

    config_file = os.path.join(os.path.split(ckpt)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    model = MPNet(h)
    state_dict = load_checkpoint(ckpt, device)
    model.load_state_dict(state_dict['generator'])
    model.forward = model.forward_export
    model.phase_decoder.forward = model.phase_decoder.forward_export
    model.eval()

    noise_amp = torch.rand(1, h.n_fft // 2 + 1, length)
    noise_pha = torch.rand(1, h.n_fft // 2 + 1, length)
    inputs = (
        noise_amp, noise_pha
    )
    input_names = ['noise_mag', 'noise_pha']
    with torch.no_grad():
        torch.onnx.export(model,               # model being run
                        inputs,                    # model input (or a tuple for multiple inputs)
                        model_name,              # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=16,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        dynamic_axes=None,
                        input_names = input_names, # the model's input names
                        output_names = ['denoise_mag', 'denoise_pha_i', 'denoise_pha_r'], # the model's output names
                        )
    sim_model,_ = simplify(model_name)
    onnx.save(sim_model, model_name)
    print(f"Export model to {model_name}")

    sess = ort.InferenceSession(model_name, providers=["CPUExecutionProvider"])

    noisy_wav, _ = librosa.load(input_audio, sr=h.sampling_rate)
    noisy_wav = torch.FloatTensor(noisy_wav).to(device)
    norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
    noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
    noisy_mag, noisy_pha = mag_pha_stft(noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    total_input_len = noisy_mag.size(-1)

    calib_path = "calibration_dataset"
    noise_mag_path = os.path.join(calib_path, "noise_mag")
    noise_pha_path = os.path.join(calib_path, "noise_pha")
    os.makedirs(calib_path, exist_ok=True)
    os.makedirs(noise_mag_path, exist_ok=True)
    os.makedirs(noise_pha_path, exist_ok=True)
    tf_mag = tf.open(os.path.join(calib_path, "noise_mag.tar.gz"), "w:gz")
    tf_pha = tf.open(os.path.join(calib_path, "noise_pha.tar.gz"), "w:gz")

    print(f"Generate calib dataset to {calib_path}")
    slice_num = int(np.ceil(total_input_len / length))
    amp_list = []
    pha_list = []
    if total_input_len < slice_num * length:
        pad_size = (0, slice_num * length - total_input_len)
        noisy_mag = torch.nn.functional.pad(noisy_mag, pad_size, mode="constant", value=0)
        noisy_pha = torch.nn.functional.pad(noisy_pha, pad_size, mode="constant", value=0)
    for i in tqdm.trange(slice_num):
        sub_mag = noisy_mag[..., i * length : (i + 1) * length]
        sub_pha = noisy_pha[..., i * length : (i + 1) * length]
        sub_mag = sub_mag.numpy()
        sub_pha = sub_pha.numpy()

        sub_mag_npy = os.path.join(noise_mag_path, f"{i}.npy")
        sub_pha_npy = os.path.join(noise_pha_path, f"{i}.npy")
        np.save(sub_mag_npy, sub_mag)
        np.save(sub_pha_npy, sub_pha)
        tf_mag.add(sub_mag_npy)
        tf_pha.add(sub_pha_npy)

        amp_g, pha_g_i, pha_g_r = sess.run(None, {"noise_mag": sub_mag, "noise_pha": sub_pha})
        amp_list.append(amp_g)
        pha_list.append(np.arctan2(pha_g_i, pha_g_r))

    tf_mag.close()
    tf_pha.close()

    amp_g = torch.from_numpy(np.concatenate(amp_list, axis=-1)[:total_input_len])
    pha_g = torch.from_numpy(np.concatenate(pha_list, axis=-1)[:total_input_len])
    audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    audio_g = audio_g / norm_factor
    audio_g = audio_g[..., :noisy_wav.size(-1)]

    sf.write(output_audio, audio_g.squeeze().cpu().numpy(), h.sampling_rate, 'PCM_16')


if __name__ == "__main__":
    main()