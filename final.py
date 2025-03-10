# import streamlit as st
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO
# import cv2
# from paddleocr import PaddleOCR  
# import pandas as pd
# import os
# import threading
# import time
# from datetime import datetime
# import re

# # Configure logging
# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Constants
# LICENSE_MODEL_DETECTION_DIR = "C:\\Users\\Dr. Shephali\\OneDrive\\Desktop\\ANpr\License-Plate-Detection-with-YoloV8-and-EasyOCR\\models\\license_plate_detector.pt"
# COCO_MODEL_DIR = "License-Plate-Detection-with-YoloV8-and-EasyOCR\\models\\yolov8n.pt"
# VEHICLES = [2]  # Class IDs for vehicles in COCO model
# RTSP_REALTIME_CSV = "rtsp_realtime_results.csv"
# CSV_OUTPUT_PATH = "final_license_plate_results.csv"
# FRAME_SKIP = 5  # ✅ Skip frames for faster processing

# # ✅ Initialize PaddleOCR
# ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')

# # ✅ Initialize YOLO models
# try:
#     coco_model = YOLO(COCO_MODEL_DIR)
#     license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)
# except Exception as e:
#     logger.error(f"Error loading YOLO models: {e}")
#     st.error("Failed to load YOLO models. Please check the model paths.")
#     st.stop()

# # ✅ Validate license plate format
# def is_valid_license_format(plate_text):
#     plate_text = ''.join(plate_text.split())
#     patterns = [
#         r'^[A-Z]{2}\d{1}[A-Z]{2}\d{4}$',  
#         r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$',  
#         r'^[A-Z]{2}\d{1}[A-Z]{3}\d{4}$'   
#     ]
#     return any(re.match(pattern, plate_text) for pattern in patterns)

# # ✅ Preprocess images for better OCR accuracy
# def preprocess_plate(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     resized = cv2.resize(gray, (400, 150))  
#     blurred = cv2.GaussianBlur(resized, (3, 3), 0)  
#     threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 15)  
#     return threshold

# # ✅ OCR Function using PaddleOCR
# def read_license_plate(license_plate_crop):
#     try:
#         preprocessed_img = preprocess_plate(license_plate_crop)
#         results = ocr_reader.ocr(preprocessed_img, cls=True)

#         best_plate, best_confidence = "", 0.0
#         for line in results:
#             for word in line:
#                 text, confidence = word[1][0], word[1][1]
#                 text = text.replace(" ", "").upper()
#                 if is_valid_license_format(text) and confidence > best_confidence:
#                     best_plate, best_confidence = text, confidence

#         return (best_plate, best_confidence) if best_plate else (None, 0.0)
#     except Exception as e:
#         logger.error(f"Error reading license plate: {e}")
#         return None, 0.0

# # ✅ Object Detection Function
# def model_prediction(img):
#     results = []
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#     # ✅ Detect vehicles
#     coco_results = coco_model(img)[0]
#     for detection in coco_results.boxes.data.tolist():
#         xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection
#         if int(class_id) in VEHICLES:
#             cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)

#     # ✅ Detect license plates
#     license_results = license_plate_detector(img)[0]
#     for license_plate in license_results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = license_plate
#         cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

#         license_plate_crop = img[int(y1):int(y2), int(x1):int(x2)]
#         license_plate_text, confidence = read_license_plate(license_plate_crop)

#         if license_plate_text:
#             results.append((license_plate_text, confidence))

#     img_with_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img_with_box, results

# # ✅ Process Video File
# def process_video_to_csv(video_path, csv_output_path):
#     cap = cv2.VideoCapture(video_path)
#     results = []
#     frame_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count % FRAME_SKIP == 0:
#             _, detected_plates = model_prediction(frame)
#             if detected_plates:
#                 timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 for plate_text, confidence in detected_plates:
#                     results.append({
#                         "timestamp": timestamp,
#                         "license_plate": plate_text,
#                         "confidence": f"{confidence:.2f}"
#                     })

#     cap.release()
#     results_df = pd.DataFrame(results)
#     results_df.to_csv(csv_output_path, index=False)
#     return results_df

# # ✅ Streamlit Interface
# def main():
#     st.title("💥 License Plate Detection 🚗")
#     options = ["Image", "Video", "RTSP Stream"]
#     choice = st.sidebar.radio("Select Detection Mode", options)

#     if choice == "Image":
#         img = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
#         if img:
#             image = np.array(Image.open(img))
#             st.image(image, caption="Uploaded Image", use_column_width=True)
#             if st.button("Detect License Plate"):
#                 prediction, detected_plates = model_prediction(image)
#                 st.image(prediction, caption="Detection Results", use_column_width=True)
#                 if detected_plates:
#                     st.success(f"Detected License Plate: {detected_plates[0][0]} (Confidence: {detected_plates[0][1]:.2f})")
#                 else:
#                     st.warning("No valid license plate detected.")

#     elif choice == "Video":
#         video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
#         if video_file:
#             with open(video_file.name, "wb") as f:
#                 f.write(video_file.getbuffer())

#             st.video(video_file.name)
#             if st.button("Process Video"):
#                 with st.spinner("Processing video..."):
#                     results_df = process_video_to_csv(video_file.name, CSV_OUTPUT_PATH)
#                     st.success("Video processing complete!")
#                     st.dataframe(results_df)
#                     with open(CSV_OUTPUT_PATH, "rb") as file:
#                         st.download_button("Download Results", file, CSV_OUTPUT_PATH, "text/csv")

#     elif choice == "RTSP Stream":
#         st.subheader("Live RTSP Stream Processing")
#         rtsp_url = st.text_input("Enter RTSP URL (e.g., rtsp://username:password@ip_address:port)")

#         if 'stream_active' not in st.session_state:
#             st.session_state.stream_active = False
#         if 'video_cap' not in st.session_state:
#             st.session_state.video_cap = None

#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("Start Stream") and not st.session_state.stream_active:
#                 st.session_state.stream_active = True
#                 st.session_state.video_cap = cv2.VideoCapture(rtsp_url)
#                 threading.Thread(target=process_rtsp_stream).start()

#         with col2:
#             if st.button("Stop Stream") and st.session_state.stream_active:
#                 st.session_state.stream_active = False
#                 if st.session_state.video_cap:
#                     st.session_state.video_cap.release()

# if __name__ == "__main__":
#     main

import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import pandas as pd
import uuid
import os
from streamlit_webrtc import webrtc_streamer
import av
import tempfile
from paddleocr import PaddleOCR
from datetime import datetime

# Set the background image
def set_background(image_path):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({image_path});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Define the paths to your model files
set_background("License-Plate-Detection-with-YoloV8-and-EasyOCR\\imgs\\test_background.jpg")
folder_path = "C:\\Users\\Dr. Shephali\\Downloads"
LICENSE_MODEL_DETECTION_DIR = "License-Plate-Detection-with-YoloV8-and-EasyOCR\\models\\license_plate_detector.pt"
COCO_MODEL_DIR = "License-Plate-Detection-with-YoloV8-and-EasyOCR\\models\\yolov8n.pt"

# Initialize the OCR reader
reader = PaddleOCR(use_angle_cls=True, lang='en')

def enhance_image(image):
    """Enhance image quality using histogram equalization and noise reduction."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    denoised = cv2.fastNlMeansDenoising(equalized, h=30)
    sharp = cv2.addWeighted(gray, 1.5, denoised, -0.5, 0)
    return sharp

def is_image_blurry(image, threshold=100):
    """Detect if the image is blurry using the Laplacian variance method."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

vehicles = [2]
coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

# Initialize Streamlit state
if "state" not in st.session_state:
    st.session_state["state"] = "Uploader"

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_to_analyze = img.copy()
        img_to_analyze = cv2.cvtColor(img_to_analyze, cv2.COLOR_RGB2BGR)
        
        if is_image_blurry(img_to_analyze, threshold=50):
            st.warning("The input image is too blurry for accurate detection.")
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        license_detections = license_plate_detector(img_to_analyze)[0]

        if len(license_detections.boxes.cls.tolist()) != 0:
            for license_plate in license_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
                license_plate_crop_gray = enhance_image(license_plate_crop)
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)
                cv2.rectangle(img, (int(x1) - 40, int(y1) - 40), (int(x2) + 40, int(y1)), (255, 255, 255), cv2.FILLED)
                cv2.putText(img, str(license_plate_text), (int((int(x1) + int(x2)) / 2) - 70, int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.ocr(license_plate_crop)
    width = img.shape[1]
    height = img.shape[0]

    if not detections:
        return None, None

    rectangle_size = license_plate_crop.shape[0] * license_plate_crop.shape[1]
    plate = []

    for result in detections:
        if not result:
            continue
        for line in result:
            bbox, text, score = line[0], line[1][0], line[1][1]
            length = np.linalg.norm(np.array(bbox[1]) - np.array(bbox[0]))
            height = np.linalg.norm(np.array(bbox[2]) - np.array(bbox[1]))

            # Filter out texts that are too short
            if 9 <= len(text) <= 11 and length * height / rectangle_size > 0.17:
                text = text.upper()
                scores += score
                plate.append(text)

    if plate:
        return " ".join(plate), scores / len(plate)
    return None, 0

def write_csv(results, file_path):
    """Writes the detection results to a CSV file."""
    if not results:
        print("No results to write to CSV.")
        return
    
    data = []
    for key, value in results.items():
        car_bbox = value['car']['bbox']
        car_score = value['car']['car_score']
        license_plate_bbox = value['license_plate']['bbox']
        license_plate_text = value['license_plate']['text']
        license_plate_bbox_score = value['license_plate']['bbox_score']
        license_plate_text_score = value['license_plate']['text_score']
        timestamp = value['timestamp']
        
        data.append([
            car_bbox, car_score,
            license_plate_bbox, license_plate_text,
            license_plate_bbox_score, license_plate_text_score,
            timestamp
        ])
    
    df = pd.DataFrame(data, columns=[
        'Car BBox', 'Car Score', 'License Plate BBox', 'License Plate Text',
        'License Plate BBox Score', 'License Plate Text Score', 'Timestamp'
    ])
    df.to_csv(file_path, index=False)
    print("Results written to CSV.")

def model_prediction(img):
    if is_image_blurry(img, threshold=50):
        st.warning("The input image is too blurry for accurate detection.")
        return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]
    
    license_numbers = 0
    results = {}
    licenses_texts = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    object_detections = coco_model(img)[0]
    license_detections = license_plate_detector(img)[0]

    if len(object_detections.boxes.cls.tolist()) != 0:
        for detection in object_detections.boxes.data.tolist():
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection
            if int(class_id) in vehicles:
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
    else:
        xcar1, ycar1, xcar2, ycar2 = 0, 0, 0, 0
        car_score = 0

    if len(license_detections.boxes.cls.tolist()) != 0:
        license_plate_crops_total = []
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
            img_name = '{}.jpg'.format(uuid.uuid1())
            cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)
            license_plate_crop_gray = enhance_image(license_plate_crop)
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)
            
            # Only store texts that pass the length check
            if license_plate_text and license_plate_text_score:
                licenses_texts.append(license_plate_text)
                license_plate_crops_total.append(license_plate_crop)
                results[license_numbers] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'car_score': car_score},
                    'license_plate': {'bbox': [x1, y1, x2, y2], 'text': license_plate_text, 'bbox_score': score,
                                      'text_score': license_plate_text_score},
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                license_numbers += 1

        write_csv(results, "License-Plate-Detection-with-YoloV8-and-EasyOCR/csv_detections/detection_results.csv")
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box, licenses_texts, license_plate_crops_total]
    else:
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box]

def process_video_to_csv(video_path, csv_output_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / frame_rate

        if is_image_blurry(frame, threshold=50):  # Adjusted threshold for more accurate blur detection
            continue

        license_detections = license_plate_detector(frame)[0]
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            license_plate_crop_gray = enhance_image(license_plate_crop)
            license_plate_text, _ = read_license_plate(license_plate_crop_gray, frame)

            if license_plate_text:
                results.append({
                    "Timestamp (s)": timestamp,
                    "License Plate": license_plate_text,
                    "Bounding Box": f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})",
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

    cap.release()

    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_output_path, index=False)
    return results_df


def connect_to_rtsp_stream(rtsp_url, csv_output_path):
    cap = cv2.VideoCapture(rtsp_url)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / frame_rate

        if is_image_blurry(frame, threshold=50):  # Adjusted threshold for more accurate blur detection
            continue

        license_detections = license_plate_detector(frame)[0]
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            license_plate_crop_gray = enhance_image(license_plate_crop)
            license_plate_text, _ = read_license_plate(license_plate_crop_gray, frame)

            if license_plate_text:
                results.append({
                    "Timestamp (s)": timestamp,
                    "License Plate": license_plate_text,
                    "Bounding Box": f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})",
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

    cap.release()

    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_output_path, index=False)
    return results_df

with st.sidebar:
    st.title("💥 License Plate Detection 🚗")
    options = ["RTSP", "Video", "Live"]
    choice = st.radio("Select Detection Mode", options)

# RTSP processing section
if choice == "RTSP":
    rtsp_url = st.text_input("Enter RTSP URL")
    if rtsp_url:
        csv_output_path = os.path.join(tempfile.gettempdir(), "results.csv")
        if st.button("Process RTSP Stream"):
            with st.spinner("Processing RTSP stream, please wait..."):
                results_df = connect_to_rtsp_stream(rtsp_url, csv_output_path)
                st.success("RTSP stream processing complete!")
                st.dataframe(results_df)
                with open(csv_output_path, "rb") as file:
                    st.download_button(label="Download Results as CSV", data=file, file_name="license_plate_results.csv",
                                       mime="text/csv")

# Video processing section
if choice == "Video":
    video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(video_file.read())
            video_path = temp_file.name

        st.video(video_path)

        csv_output_path = os.path.join(tempfile.gettempdir(), "results.csv")
        if st.button("Process Video"):
            with st.spinner("Processing video, please wait..."):
                results_df = process_video_to_csv(video_path, csv_output_path)
                st.success("Video processing complete!")
                st.dataframe(results_df)
                with open(csv_output_path, "rb") as file:
                    st.download_button(label="Download Results as CSV", data=file, file_name="license_plate_results.csv",
                                       mime="text/csv")

# State change functions
def change_state_uploader():
    st.session_state["state"] = "Uploader"

def change_state_camera():
    st.session_state["state"] = "Camera"

def change_state_live():
    st.session_state["state"] = "Live"

# Header section
with st.container() as header:
    _, col1, _ = st.columns([0.2, 1, 0.1])
    col1.title("💥 License Car Plate Detection 🚗")

    _, col0, _ = st.columns([0.15, 1, 0.1])
    col0.image("License-Plate-Detection-with-YoloV8-and-EasyOCR\\imgs\\test_background.jpg", width=500)

    _, col4, _ = st.columns([0.1, 1, 0.2])
    col4.subheader("Computer Vision Detection with YoloV8 🧪")

    _, col, _ = st.columns([0.3, 1, 0.1])
    col.image("License-Plate-Detection-with-YoloV8-and-EasyOCR\\imgs\\test_background.jpg")

    _, col5, _ = st.columns([0.05, 1, 0.1])
    st.write("The models detect the car and the license plate in a given image, then extract the info using PaddleOCR, and crop and save the license plate as an image, with a CSV file containing all the data.")

# Body section
with st.container() as body:
    _, col1, _ = st.columns([0.1, 1, 0.2])
    col1.subheader("Check out the License Car Plate Detection Model 🔎!")

    _, colb1, colb2, colb3 = st.columns([0.2, 0.7, 0.6, 1])

    if colb1.button("Upload an Image"):
        st.session_state["state"] = "Uploader"
    elif colb2.button("Take a Photo"):
        st.session_state["state"] = "Camera"
    elif colb3.button("Live Detection"):
        st.session_state["state"] = "Live"

    img = None
    if st.session_state["state"] == "Uploader":
        img = st.file_uploader("Upload a Car Image: ", type=["png", "jpg", "jpeg"])
    elif st.session_state["state"] == "Camera":
        img = st.camera_input("Take a Photo: ")
    elif st.session_state["state"] == "Live":
        webrtc_streamer(key="sample", video_processor_factory=VideoProcessor)

    if img is not None:
        image = np.array(Image.open(img))
        _, col2, _ = st.columns([0.3, 1, 0.2])
        col2.image(image, width=400)

        if st.button("Apply Detection"):
            results = model_prediction(image)
            if len(results) == 3:
                prediction, texts, license_plate_crop = results[0], results[1], results[2]
                texts = [i for i in texts if i is not None]
                print("Detected texts:", texts)

                if len(texts) == 1 and len(license_plate_crop):
                    _, col3, _ = st.columns([0.4, 1, 0.2])
                    col3.header("Detection Results ✅:")
                    _, col4, _ = st.columns([0.1, 1, 0.1])
                    col4.image(prediction)
                    _, col9, _ = st.columns([0.4, 1, 0.2])
                    col9.header("License Cropped ✅:")
                    _, col10, _ = st.columns([0.3, 1, 0.1])
                    col10.image(license_plate_crop[0], width=350)
                    _, col11, _ = st.columns([0.45, 1, 0.55])
                    col11.success(f"License Number: {texts[0]}")
                    df = pd.read_csv("License-Plate-Detection-with-YoloV8-and-EasyOCR/csv_detections/detection_results.csv")
                    st.dataframe(df)
                elif len(texts) > 1 and len(license_plate_crop) > 1:
                    _, col3, _ = st.columns([0.4, 1, 0.2])
                    col3.header("Detection Results ✅:")
                    _, col4, _ = st.columns([0.1, 1, 0.1])
                    col4.image(prediction)
                    _, col9, _ = st.columns([0.4, 1, 0.2])
                    col9.header("License Cropped ✅:")
                    _, col10, _ = st.columns([0.3, 1, 0.1])

                    for i in range(0, len(license_plate_crop)):
                        col10.image(license_plate_crop[i], width=350)
                        _, col11, _ = st.columns([0.45, 1, 0.55])
                        col11.success(f"License Number {i}: {texts[i]}")

                    df = pd.read_csv("License-Plate-Detection-with-YoloV8-and-EasyOCR/csv_detections/detection_results.csv")
                    st.dataframe(df)
            else:
                prediction = results[0]
                _, col3, _ = st.columns([0.4, 1, 0.2])
                col3.header("Detection Results ✅:")
                _, col4, _ = st.columns([0.3, 1, 0.1])
                col4.image(prediction)