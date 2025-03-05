# BatchID Engine 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7-red)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25-green)](https://streamlit.io)

**Smart Batch ID Photo Processor**​ - Automate professional ID photo generation with customizable parameters

## 📌 Description

BatchID Engine is an intelligent batch processing system for standardized ID photo generation. It automates background replacement, facial alignment and size calibration while preserving original directory structures. Key features include:

✅ ​**Smart Cropping**​ - AI-powered facial detection and alignment  
✅ ​**Custom Presets**​ - Dynamic size/background/margin controls  
✅ ​**Error Resilient**​ - Fault-tolerant pipeline with error logging

## 🎯 Features

### 1. Intelligent Batch Processing
- Process nested ZIP archives (max 500MB)
- Preserve original folder structure
- Auto-skip non-image files

### 2. Professional Parameter Controls
| Feature         | Range             | Default   |
|-----------------|-------------------|-----------|
| ​**Resolution**​  | 600x800~2400x3000| 1200x1500 |
| ​**Head Margin**​ | 50~500px         | 250px     |
| ​**Background**​  | Auto/Custom Color| Auto      |

### 3. Technical Highlights
- OpenCV DNN face detection
- Adaptive background recognition
- Edge-aware cropping algorithm
- Session-based error tracking

## 🛠️ Deployment

### Prerequisites
- Python 3.8+

### Quick Start
```bash
# Clone repo
git clone https://github.com/kevinmattew/BatchID-Engine.git
cd batchid-engine

# Install dependencies
pip install -r requirements.txt

# Download AI models
wget https://github.com/opencv/opencv/raw/4.x/samples/dnn/face_detector/deploy.prototxt
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel

# Run with upload limit
streamlit run app.py --server.maxUploadSize=[Filesize mb]
