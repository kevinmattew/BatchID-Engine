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

def calculate_head_position(face_box, img):
    """估算头部顶部位置"""
    (startX, startY, endX, endY) = face_box
    face_height = endY - startY
    
    # 在面部区域上方提取ROI进行边缘检测
    roi_y1 = max(0, startY - int(face_height * 1.5))
    roi = img[roi_y1:startY, startX:endX]
    
    # 灰度转换和边缘检测
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # 寻找最上方的连续边缘
    edge_rows = np.where(edges > 0)[0]
    if len(edge_rows) > 0:
        head_top_rel = edge_rows[0]
        return roi_y1 + head_top_rel
    else:
        return startY - int(face_height * 0.3)  # 默认估计

def smart_crop(img):
    """智能裁剪核心逻辑"""
    (h, w) = img.shape[:2]
    
    # 人脸检测
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                               (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # 选择最佳人脸
    best_conf = 0
    best_box = None
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.9 and confidence > best_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box = box.astype("int")
            best_conf = confidence

    if best_box is None:
        return None

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
            # 保持宽高比缩放原图并添加背景
            h_img, w_img = img.shape[:2]
            target_aspect = target_w / target_h
            current_aspect = w_img / h_img

            if current_aspect > target_aspect:
                new_w = target_w
                new_h = int(h_img * (target_w / w_img))
            else:
                new_h = target_h
                new_w = int(w_img * (target_h / h_img))
            
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            result = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        else:
            (startX, startY, endX, endY) = best_box
            head_top = calculate_head_position(best_box, img)
            
            # 动态裁剪逻辑
            crop_top = max(0, head_top - head_margin)
            required_height = int(target_h * (w / target_w))
            crop_bottom = min(h, crop_top + required_height)
            
            # 边界处理：扩展底部背景
            if crop_bottom > h:
                img = cv2.copyMakeBorder(img, 0, crop_bottom-h, 0, 0,
                                       cv2.BORDER_CONSTANT, value=bg_color)
                h = crop_bottom  # 更新图片高度
            
            cropped = img[crop_top:crop_bottom, :]
            
            # 保持宽高比缩放并添加背景
            h_cropped, w_cropped = cropped.shape[:2]
            target_aspect = target_w / target_h
            current_aspect = w_cropped / h_cropped

            if current_aspect > target_aspect:
                new_w = target_w
                new_h = int(h_cropped * (target_w / w_cropped))
            else:
                new_h = target_h
                new_w = int(w_cropped * (target_h / h_cropped))
            
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
            result = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result)
        return True, "处理成功"
    except Exception as e:
        return False, str(e)

def process_zip(uploaded_zip, target_size, head_margin, bg_mode):
    with tempfile.TemporaryDirectory() as tmp_in:
        # 解压上传的ZIP并处理文件名编码
        zf = zipfile.ZipFile(uploaded_zip)
        for file_info in zf.infolist():
            # 解码文件名
            try:
                decoded_name = file_info.filename.encode('cp437').decode('utf-8')
            except UnicodeDecodeError:
                try:
                    decoded_name = file_info.filename.encode('cp437').decode('gbk')
                except:
                    decoded_name = file_info.filename  # 保留原始名称
            
            target_path = os.path.join(tmp_in, decoded_name)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            if not file_info.is_dir():
                with open(target_path, 'wb') as f:
                    f.write(zf.read(file_info))

        results = {'total':0, 'success':0, 'errors':[]}
        output_buffer = io.BytesIO()
        
        # 创建支持UTF-8编码的ZIP文件
        with zipfile.ZipFile(output_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as output_zip:
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
                            # 添加文件到ZIP并确保UTF-8编码
                            zip_info = zipfile.ZipInfo(relative_path)
                            zip_info.flag_bits |= 0x800  # 设置UTF-8标志位
                            with open(output_path, 'rb') as f:
                                data = f.read()
                            output_zip.writestr(zip_info, data)
                            results['success'] += 1
                        else:
                            error_msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {relative_path}: {message}"
                            results['errors'].append(error_msg)
                            st.session_state.error_history.append(error_msg)
        
        output_buffer.seek(0)
        return results, output_buffer

# Streamlit界面
st.title("📸 智能证件照处理系统")
st.markdown("自动生成1200×1500证件照，包含蓝色背景")

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
