import cv2
import os

def mp4_to_png(video_path, output_folder):
    """
    Converts an MP4 video into a series of PNG images.
    
    Args:
        video_path (str): Path to the input MP4 video.
        output_folder (str): Path to the folder where PNG images will be saved.
    """
    # Check if output folder exists, create if it doesn't
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    frame_count = 0
    
    while True:
        # Read each frame
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no frame is returned
        
        # Save the frame as a PNG image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved: {frame_filename}")
        
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    print(f"Conversion complete! Total frames: {frame_count}")

# Usage example
if __name__ == "__main__":
    video_path = "aiforce.mp4"  # Path to your input MP4 video
    output_folder = "aiforce_11_15"  # Folder to store the PNG images
    mp4_to_png(video_path, output_folder)
