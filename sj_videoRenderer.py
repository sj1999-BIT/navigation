import cv2
import os

def images_to_video(image_folder, output_file, fps=30):
    """
    Convert a sequence of images in a folder to a video file, using directory order.
    
    Args:
        image_folder (str): Path to folder containing image sequence
        output_file (str): Path where the video will be saved (include .mp4 extension)
        fps (int): Frames per second for output video
    """
    try:
        # Get list of image files
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        if not images:
            print(f"No images found in {image_folder}")
            return
            
        # Read first image to get dimensions
        first_image_path = os.path.join(image_folder, images[0])
        frame = cv2.imread(first_image_path)
        height, width = frame.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Process each image
        for image_file in images:
            image_path = os.path.join(image_folder, image_file)
            frame = cv2.imread(image_path)
            video_writer.write(frame)
            
        # Release video writer
        video_writer.release()
        print(f"Video saved to: {output_file}")
        
    except Exception as e:
        print(f"Error creating video: {str(e)}")

def create_videos_from_observations(base_folder="./test", fps=30):
    """Create videos from all observation types"""
    folders = ["rgba", "depth"]
    
    for folder in folders:
        input_folder = os.path.join(base_folder, folder)
        if os.path.exists(input_folder):
            output_file = os.path.join(base_folder, f"agent_view_{folder}.mp4")
            print(f"\nProcessing {folder} images...")
            images_to_video(input_folder, output_file, fps)