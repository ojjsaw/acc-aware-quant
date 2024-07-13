# acc-aware-quant

#### Setup
```sh
sudo apt-get update
sudo apt-get install libgl1-mesa-glx
pip install "openvino>=2024.0.0" "nncf>=2.9.0"
pip install "torch>=2.1" "torchvision>=0.16" "ultralytics==8.2.24" onnx tqdm opencv-python --extra-index-url https://download.pytorch.org/whl/cpu
```

#### Dataset Helpers
```sh
mkdir datasets
unzip archive.zip -d datasets
unzip seatbelt.v3i.yolov8.zip -d datasets
```