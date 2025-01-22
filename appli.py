import streamlit as st
import cv2
import os
import tempfile
from util import set_background, process_video, write_csv

# Set the background for the Streamlit app
set_background("background_image.png")  # Replace with your background image path

# Streamlit app title and description
st.title("License Plate Detection App")
st.write("Upload a video to detect license plates in vehicles.")

# Upload video file
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(temp_video_path, "wb") as temp_file:
        temp_file.write(uploaded_video.read())

    st.video(temp_video_path)  # Display the uploaded video

    # Process video on button click
    if st.button("Process Video"):
        st.write("Processing video, please wait...")

        # Call process_video function from util.py
        results = process_video(temp_video_path)

        # Save results to a CSV file
        output_csv_path = os.path.join(os.getcwd(), "detection_results.csv")
        write_csv(results, output_csv_path)

        st.success("Processing complete! Results saved in 'detection_results.csv'")

        # Option to download the CSV file
        with open(output_csv_path, "rb") as csv_file:
            st.download_button(
                label="Download Detection Results as CSV",
                data=csv_file,
                file_name="detection_results.csv",
                mime="text/csv",
            )

else:
    st.write("Please upload a video to begin processing.")
