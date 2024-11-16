import os
import math
import shutil

def split_images_into_folders(input_folder, output_folder_prefix="output_folder", num_splits=5):
    """
    Split image files from one folder into multiple folders, dividing them evenly.
    
    Args:
        input_folder (str): Path to the folder containing the images.
        output_folder_prefix (str): Prefix for the output folders (e.g., "output_folder").
        num_splits (int): Number of folders to split the images into.
    """
    # Get a sorted list of all files in the input folder
    files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(".png")])
    
    if not files:
        print("No PNG files found in the input folder.")
        return
    
    # Calculate the number of files per folder
    total_files = len(files)
    files_per_split = math.ceil(total_files / num_splits)  # Handle cases where files can't be evenly divided
    
    print(f"Total files: {total_files}")
    print(f"Files per folder: {files_per_split}")
    
    # Split the files and move them to separate folders
    for i in range(num_splits):
        # Create a new folder for this split
        split_folder = f"{output_folder_prefix}_{i+1}"
        os.makedirs(split_folder, exist_ok=True)
        
        # Get the files for this split
        start_index = i * files_per_split
        end_index = min(start_index + files_per_split, total_files)  # Ensure we don't go out of bounds
        split_files = files[start_index:end_index]
        
        # Move the files to the split folder
        for file_name in split_files:
            src_path = os.path.join(input_folder, file_name)
            dst_path = os.path.join(split_folder, file_name)
            shutil.copy(src_path, dst_path)
        
        print(f"Created folder: {split_folder} with {len(split_files)} files.")
    
    print("Splitting complete!")

# 사용 예제
if __name__ == "__main__":
    input_folder = "aiforce_11_15"  # 정리할 이미지가 있는 폴더
    split_images_into_folders(input_folder, 5)