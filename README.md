# EAST

This is a tensorflow re-implementation for the paper: "EAST: An Efficient and Accurate Scene Text Detector"

More details: https://arxiv.org/pdf/1704.03155.pdf

# Dependencies
* Python3
* tensorflow
* numpy
* opencv-python
* shapely

Dependencies can be installed with
```bash
pip install -r requirements.txt
```

# Data Preparation
* Put all images in ./data/img/
* Put corresponding ground truth in ./data/gt

For example:
```bash
./data/img/1.jpg
./data/gt/1.txt
```
data format: [Link](https://tianchi.aliyun.com/competition/entrance/231651/information)

# Preprocess
```bash
python data_processor.py
```

# Train
```bash
python train.py
```

# Valid
```bash
python valid.py
```