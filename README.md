# mpsenet.axera
MP-SENet speech enhancement model on Axera  

  
**预编译模型已在本仓库中，如需自行转换请参考以下步骤。**

## 转换模型

### 环境准备
```
conda create -n mpsenet --file requirements_export.txt
conda activate mpsenet
```

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
pip3 install -r requirements_ax.txt
```
```
python3 run_ax.py -i noisy_snr0.wav
```

```
root@ax650:/mnt/rzyang/Codes/mpsenet.axera# python3 run_ax.py -i noisy_snr0.wav
Load model take 228.17015647888184ms
100%|███████████████████████████████████████████████████████████████████████████████| 14/14 [00:01<00:00, 13.09it/s]
Save output audio to clean.wav
Average inference time: 73.51902553013393ms
RTF: 0.09714012020662888
```
