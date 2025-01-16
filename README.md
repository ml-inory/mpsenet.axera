# mpsenet.axera
MP-SENet speech enhancement model on Axera

## 环境准备
```
conda create -n mpsenet --file requirements.txt
conda activate mpsenet
```

## 转换模型

### ONNX
```
python export_onnx.py
```

### axmodel
```
pulsar2 build --input mp-senet.onnx --config config_mpsenet.json --output_dir axmodel --output_name mp-senet.axmodel --target_hardware AX650 --npu_mode NPU3 --compiler.check 0
```

## 运行

### ONNX
```
python run_ort.py -i noisy_snr0.wav
```

### axmodel
```
python3 run_ax.py -i noisy_snr0.wav
```