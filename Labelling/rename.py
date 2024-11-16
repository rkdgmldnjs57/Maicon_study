import os

def rename_images_in_folder(folder_path, prefix="frame_"):
    """
    Rename image files in a folder to have sequential names (e.g., frame_0000.png, frame_0001.png).
    
    Args:
        folder_path (str): Path to the folder containing the images.
        prefix (str): Prefix for the renamed images.
    """
    # Get a sorted list of all files in the folder
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".png")])
    
    if not files:
        print("No PNG files found in the folder.")
        return
    
    for idx, filename in enumerate(files):
        # Generate new name with zero-padded numbering
        new_name = f"{prefix}{idx:04d}.png"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
    
    print("Renaming complete!")

# 사용 예제
if __name__ == "__main__":
    folder_path = "aiforce_11_15"  # 이미지가 저장된 폴더 경로
    
    rename_images_in_folder(folder_path)
