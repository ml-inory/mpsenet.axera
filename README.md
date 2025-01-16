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
Load model take 43.61438751220703ms
  0%|                                                                                        | 0/14 [00:00<?, ?it/s]Run model take 73.8382339477539ms
Run model take 73.47893714904785ms
 14%|███████████▍                                                                    | 2/14 [00:00<00:00, 13.11it/s]Run model take 73.7910270690918ms
Run model take 73.44603538513184ms
 29%|██████████████████████▊                                                         | 4/14 [00:00<00:00, 13.08it/s]Run model take 73.43530654907227ms
Run model take 73.34518432617188ms
 43%|██████████████████████████████████▎                                             | 6/14 [00:00<00:00, 13.08it/s]Run model take 73.47726821899414ms
Run model take 73.7009048461914ms
 57%|█████████████████████████████████████████████▋                                  | 8/14 [00:00<00:00, 13.07it/s]Run model take 73.40693473815918ms
Run model take 73.46343994140625ms
 71%|████████████████████████████████████████████████████████▍                      | 10/14 [00:00<00:00, 13.08it/s]Run model take 73.45080375671387ms
Run model take 73.44818115234375ms
 86%|███████████████████████████████████████████████████████████████████▋           | 12/14 [00:00<00:00, 13.08it/s]Run model take 73.86112213134766ms
Run model take 73.36902618408203ms
100%|███████████████████████████████████████████████████████████████████████████████| 14/14 [00:01<00:00, 13.07it/s]
Save output audio to clean.wav

```
