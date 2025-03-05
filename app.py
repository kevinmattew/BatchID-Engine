import io
import os
import cv2
import numpy as np
import streamlit as st
import time
import zipfile
import tempfile
from pathlib import Path
from collections import defaultdict

# 初始化session状态
if 'error_history' not in st.session_state:
    st.session_state.error_history = []
# 模型路径检查
PROTOTXT_PATH = "deploy.prototxt"
CAFFEMODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(CAFFEMODEL_PATH):
    st.error("缺少必要模型文件！请确保存在：deploy.prototxt 和 res10_300x300_ssd_iter_140000.caffemodel")
    st.stop()

try:
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
except Exception as e:
    st.error(f"模型加载失败: {str(e)}")
    st.stop()

# 侧边栏参数设置
with st.sidebar:
    st.header("⚙️ 参数配置")
    
    # 尺寸设置
    col1, col2 = st.columns(2)
    with col1:
        target_width = st.number_input("目标宽度", 600, 2400, 1200, step=50)
    with col2:
        target_height = st.number_input("目标高度", 800, 3000, 1500, step=50)
    
    # 头部间距
    head_margin = st.slider("头部上方间距(px)", 50, 500, 200)
    
    # 背景设置
    bg_mode = st.radio("背景模式", ["自动检测", "自定义颜色"])
    custom_bg = (255, 255, 255)
    if bg_mode == "自定义颜色":
        bg_color = st.color_picker("选择背景色", "#FFFFFF")
        custom_bg = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))[::-1]  # RGB转BGR

# 调试函数
def debug_log(message):
    if st.session_state.get('debug_mode', False):
        st.write(f"[DEBUG] {message}")

# 改进的背景检测
def detect_background_color(img, custom_bg=None):
    try:
        if custom_bg is not None:
            return custom_bg
        
        h, w = img.shape[:2]
        border_size = max(5, int(min(h,w)*0.05))
        regions = [
            img[:border_size, :], 
            img[-border_size:, :],
            img[:, :border_size], 
            img[:, -border_size:]
        ]
        pixels = np.vstack([r.reshape(-1,3) for r in regions if r.size > 0])
        
        if len(pixels) > 0:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels.astype(np.float32), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            bg_color = centers[np.argmax(np.bincount(labels.flatten()))]
            return tuple(map(int, bg_color))
        return (255, 255, 255)
    except:
        return (255, 255, 255)

def safe_resize(image, target_size):
    """带尺寸校验的缩放"""
    if image.size == 0:
        return None
    try:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        debug_log(f"缩放失败: {str(e)}")
        return None

def calculate_head_position(face_box, img):
    """带异常保护的头部定位"""
    try:
        (startX, startY, endX, endY) = face_box
        face_h = endY - startY
        
        # 计算检测区域
        roi_top = max(0, startY - int(face_h * 1.2))
        roi = img[roi_top:startY, startX:endX]
        
        if roi.size == 0:
            return startY - int(face_h * 0.25)
        
        # 边缘检测优化
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 寻找有效边缘
        edge_rows = np.where(edges > 0)[0]
        if len(edge_rows) > 0:
            return roi_top + edge_rows[0]
        return startY - int(face_h * 0.25)
    except Exception as e:
        debug_log(f"头部定位异常: {str(e)}")
        return startY  # 安全回退到面部顶部

# 修改后的核心处理函数
def process_image_file(input_path, output_path, target_size, head_margin, bg_mode):
    try:
        img = cv2.imread(input_path)
        if img is None:
            return False, "无法读取图片文件"
        
        # 获取背景颜色
        custom_bg = (255, 255, 255)
        if bg_mode == "自定义颜色":
            custom_bg = globals().get('custom_bg', (255, 255, 255))
        bg_color = detect_background_color(img, custom_bg if bg_mode == "自定义颜色" else None)

        (h, w) = img.shape[:2]
        target_w, target_h = target_size
        
        # 人脸检测
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, 
                                    (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        best_conf, best_box = 0.0, None
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > max(best_conf, 0.7):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                best_box = box.astype("int")
                best_conf = confidence

        if best_box is None:
            debug_log(f"未检测到人脸: {input_path}")
            result = cv2.resize(img, (target_w, target_h))
        else:
            (startX, startY, endX, endY) = best_box
            head_top = calculate_head_position(best_box, img)
            
            # 动态裁剪逻辑
            crop_top = max(0, head_top - head_margin)
            required_height = int(target_h * (w / target_w))
            crop_bottom = min(h, crop_top + required_height)
            
            # 边界处理
            if crop_bottom > h:
                img = cv2.copyMakeBorder(img, 0, crop_bottom-h, 0, 0,
                                       cv2.BORDER_CONSTANT, value=bg_color)
            
            cropped = img[crop_top:crop_bottom, :]
            
            # 最终合成
            resized = safe_resize(cropped, (target_w, target_h))
            if resized is None:
                resized = cv2.resize(img, (target_w, target_h))
            
            result = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
            result = cv2.addWeighted(result, 0, resized, 1, 0)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result)
        return True, "处理成功"
    except Exception as e:
        return False, str(e)

# 修改后的主处理函数
def process_zip(uploaded_zip, target_size, head_margin, bg_mode):
    with tempfile.TemporaryDirectory() as tmp_in:
        # 解压上传的ZIP
        with zipfile.ZipFile(uploaded_zip) as zf:
            zf.extractall(tmp_in)
        
        results = {'total':0, 'success':0, 'errors':[]}
        output_buffer = io.BytesIO()
        
        with zipfile.ZipFile(output_buffer, 'w') as output_zip:
            for root, _, files in os.walk(tmp_in):
                for filename in files:
                    if filename.lower().split('.')[-1] not in ['jpg', 'jpeg', 'png']:
                        continue
                    
                    input_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(input_path, tmp_in)
                    results['total'] += 1
                    
                    with tempfile.TemporaryDirectory() as process_tmp:
                        output_path = os.path.join(process_tmp, "processed.jpg")
                        success, message = process_image_file(
                            input_path, 
                            output_path,
                            target_size=target_size,
                            head_margin=head_margin,
                            bg_mode=bg_mode
                        )
                        
                        if success:
                            output_zip.write(output_path, arcname=relative_path)
                            results['success'] += 1
                        else:
                            error_msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {relative_path}: {message}"
                            results['errors'].append(error_msg)
                            st.session_state.error_history.append(error_msg)
        
        output_buffer.seek(0)
        return results, output_buffer

# Streamlit界面
st.title("📁 智能证件照批量处理系统")
st.checkbox("调试模式", key='debug_mode')

uploaded_zip = st.file_uploader("上传ZIP压缩包", type=["zip"])

# 实时预览功能
with st.expander("🖼 实时预览", expanded=True):
    preview_file = st.file_uploader("上传预览图片", type=["jpg", "png", "jpeg"])
    if preview_file:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "preview.jpg")
            output_path = os.path.join(tmp_dir, "preview_out.jpg")
            
            with open(input_path, "wb") as f:
                f.write(preview_file.getbuffer())
            
            success, _ = process_image_file(
                input_path, output_path,
                target_size=(target_width, target_height),
                head_margin=head_margin,
                bg_mode=bg_mode
            )
            
            if success:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.imread(input_path)[:, :, ::-1], 
                           caption="原始图片", use_column_width=True)
                with col2:
                    st.image(cv2.imread(output_path)[:, :, ::-1],
                           caption="处理效果", use_column_width=True)
            else:
                st.error("预览处理失败")

if uploaded_zip:
    if st.button("🚀 开始批量处理"):
        with st.spinner(f'正在处理 {uploaded_zip.name}...'):
            results, output_buffer = process_zip(
                uploaded_zip,
                target_size=(target_width, target_height),
                head_margin=head_margin,
                bg_mode=bg_mode
            )
            
        st.success(f"""
        ✅ 处理完成！  
        成功: {results['success']} 张  
        失败: {len(results['errors'])} 张
        """)
        
        if results['errors']:
            with st.expander("❌ 错误详情", expanded=False):
                st.code("\n".join(results['errors']))
        
        st.download_button(
            label="📥 下载处理结果",
            data=output_buffer,
            file_name="processed_photos.zip",
            mime="application/zip",
            type="primary"
        )