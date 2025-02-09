import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
import os
import tempfile
import re
import logging
import threading
import time
from datetime import datetime
from util_main import (
    set_background, 
    write_csv,
    license_complies_format,  # Add this
    format_license            # Add this
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
LICENSE_MODEL_DETECTION_DIR = "License-Plate-Detection-with-YoloV8-and-EasyOCR/models/license_plate_detector.pt"
COCO_MODEL_DIR = "License-Plate-Detection-with-YoloV8-and-EasyOCR/models/yolov8n.pt"
VEHICLES = [2]  # Class IDs for vehicles in COCO model
RTSP_REALTIME_CSV = "rtsp_realtime_results.csv"
CSV_OUTPUT_PATH = "final_license_plate_results.csv"

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Initialize YOLO models
try:
    coco_model = YOLO(COCO_MODEL_DIR)
    license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)
except Exception as e:
    logger.error(f"Error loading YOLO models: {e}")
    st.error("Failed to load YOLO models. Please check the model paths.")
    st.stop()

# Function to validate license plate format
def is_valid_license_format(plate_text):
    plate_text = ''.join(plate_text.split())
    format3 = r'^[A-Z]{2}\d{1}[A-Z]{2}\d{4}$'  # AB1CD2345
    format1 = r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'  # AB12CD3456
    format2 = r'^[A-Z]{2}\d{1}[A-Z]{3}\d{4}$'  # AB1CDE2345
    return bool(re.match(format1, plate_text) or re.match(format2, plate_text)) or re.match(format3, plate_text)

# Function to preprocess license plate images for OCR
def preprocess_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (300, 100))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    return threshold



def read_license_plate(license_plate_crop):
    try:
        MIN_CONFIDENCE = 0.05
        preprocessed_img = preprocess_plate(license_plate_crop)
        detections = reader.readtext(preprocessed_img)

        if not detections:
            return None, 0.0

        best_plate = ""
        best_confidence = 0.0
        for bbox, text, score in detections:
            text = text.upper().replace(" ", "")
            if is_valid_license_format(text) and score >= MIN_CONFIDENCE and score > best_confidence:
                best_plate = text
                best_confidence = score

        return (best_plate, best_confidence) if best_plate else (None, 0.0)
    except Exception as e:
        logger.error(f"Error reading license plate: {e}")
        return None, 0.0


def model_prediction(img):
    results = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Detect vehicles
    coco_results = coco_model(img)[0]
    for detection in coco_results.boxes.data.tolist():
        xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection
        if int(class_id) in VEHICLES:
            cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)

    # Detect license plates
    license_results = license_plate_detector(img)[0]
    for license_plate in license_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

        # Crop license plate region
        license_plate_crop = img[int(y1):int(y2), int(x1):int(x2)]
        license_plate_text, confidence = read_license_plate(license_plate_crop)

        if license_plate_text:
            results.append((license_plate_text, confidence))

    img_with_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_with_box, results

def main():
    st.title("💥 License Plate Detection 🚗")
    options = ["Image", "Video", "RTSP Stream"]
    choice = st.sidebar.radio("Select Detection Mode", options)

    if choice == "Video":
        video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        if video_file:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(video_file.read())
                video_path = temp_file.name

            st.video(video_path)
            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    results_df = process_video_to_csv(video_path, CSV_OUTPUT_PATH)
                    st.success("Video processing complete!")
                    st.dataframe(results_df)
                    with open(CSV_OUTPUT_PATH, "rb") as file:
                        st.download_button("Download Results", file, CSV_OUTPUT_PATH, "text/csv")

    elif choice == "Image":
        img = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
        if img:
            image = np.array(Image.open(img))
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Detect License Plate"):
                prediction, detected_plates = model_prediction(image)
                st.image(prediction, caption="Detection Results", use_column_width=True)
                if detected_plates:
                    st.success(f"Detected License Plate: {detected_plates[0][0]} (Confidence: {detected_plates[0][1]:.2f})")
                else:
                    st.warning("No valid license plate detected.")

    elif choice == "RTSP Stream":
        st.subheader("Live RTSP Stream Processing")
        rtsp_url = st.text_input("Enter RTSP URL (e.g., rtsp://username:password@ip_address:port)")
        
        # Initialize session state
        if 'stream_active' not in st.session_state:
            st.session_state.stream_active = False
        if 'realtime_results' not in st.session_state:
            st.session_state.realtime_results = []
        if 'final_results' not in st.session_state:
            st.session_state.final_results = []
        if 'video_cap' not in st.session_state:
            st.session_state.video_cap = None
        if 'latest_frame' not in st.session_state:
            st.session_state.latest_frame = None
        if 'csv_update_time' not in st.session_state:
            st.session_state.csv_update_time = time.time()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Stream") and not st.session_state.stream_active:
                st.session_state.stream_active = True
                st.session_state.video_cap = cv2.VideoCapture(rtsp_url)
                st.session_state.realtime_results = []
                st.session_state.final_results = []
                st.session_state.process_thread = threading.Thread(target=process_rtsp_stream)
                st.session_state.process_thread.start()

        with col2:
            if st.button("Stop Stream") and st.session_state.stream_active:
                st.session_state.stream_active = False
                if st.session_state.video_cap is not None:
                    st.session_state.video_cap.release()
                st.session_state.video_cap = None
                save_results_to_csv(st.session_state.final_results, CSV_OUTPUT_PATH)
                st.success(f"Final results saved to {CSV_OUTPUT_PATH}")

        # Display live feed and results
        if st.session_state.stream_active:
            frame_placeholder = st.empty()
            results_placeholder = st.empty()
            download_placeholder = st.empty()

            while st.session_state.stream_active:
                if st.session_state.latest_frame is not None:
                    frame_placeholder.image(st.session_state.latest_frame, channels="RGB")
                if st.session_state.realtime_results:
                    results_placeholder.dataframe(pd.DataFrame(st.session_state.realtime_results[-10:]))
                
                # Auto-save real-time CSV every 5 seconds
                if time.time() - st.session_state.csv_update_time > 5:
                    save_results_to_csv(st.session_state.realtime_results, RTSP_REALTIME_CSV)
                    st.session_state.csv_update_time = time.time()
                    with open(RTSP_REALTIME_CSV, "rb") as file:
                        download_placeholder.download_button(
                            "Download Real-time Results",
                            data=file,
                            file_name=RTSP_REALTIME_CSV,
                            mime="text/csv"
                        )
                
                time.sleep(0.1)

        # Final download button after stopping
        if not st.session_state.stream_active and st.session_state.final_results:
            with open(CSV_OUTPUT_PATH, "rb") as file:
                st.download_button("Download Final Results", file, CSV_OUTPUT_PATH, "text/csv")

def save_results_to_csv(results, filename):
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(filename, index=False)
        logger.info(f"CSV {filename} updated with {len(results)} records")

def process_rtsp_stream():
    while st.session_state.stream_active and st.session_state.video_cap.isOpened():
        ret, frame = st.session_state.video_cap.read()
        if not ret:
            st.error("Failed to read frame from RTSP stream")
            break
        
        if int(time.time() * 1000) % 5 == 0:
            processed_frame, plates = model_prediction(frame)
            st.session_state.latest_frame = processed_frame
            
            if plates:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for plate_text, confidence in plates:
                    record = {
                        "timestamp": timestamp,
                        "license_plate": plate_text,
                        "confidence": f"{confidence:.2f}"
                    }
                    st.session_state.realtime_results.append(record)
                    st.session_state.final_results.append(record)

def process_video_to_csv(video_path, csv_output_path):
    cap = cv2.VideoCapture(video_path)
    results = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            _, detected_plates = model_prediction(frame)
            if detected_plates:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for plate_text, confidence in detected_plates:
                    results.append({
                        "timestamp": timestamp,
                        "license_plate": plate_text,
                        "confidence": f"{confidence:.2f}"
                    })

    cap.release()
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_output_path, index=False)
    return results_df

if __name__ == "__main__":
    main()
