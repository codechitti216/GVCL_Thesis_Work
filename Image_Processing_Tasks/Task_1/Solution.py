import numpy as np
import cv2
from scipy.stats import norm

# Load the video
cap = cv2.VideoCapture('Video.mp4')

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

vertical_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
horizontal_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])


def apply_kernel(img_array, horizontal_kernel, vertical_kernel):
    threshold = 40
    
    new_img = np.zeros_like(img_array)  # Initialize new_img with zeros
    for i in range(1, len(img_array) - 1):
        for j in range(1, len(img_array[0]) - 1):
            h_sum = np.sum(img_array[i-1:i+2, j-1:j+2] * horizontal_kernel)
            v_sum = np.sum(img_array[i-1:i+2, j-1:j+2] * vertical_kernel)
            value = int(np.sqrt((h_sum**2) + (v_sum**2)))
            if value <= threshold:
                new_img[i][j] = 255
            else:
                new_img[i][j] = 0
    
    return new_img

def gaussian_distribution(img_array, lower_prob, upper_prob):
    # Flatten the image array to get all pixel values
    pixel_values = img_array.flatten()
    
    # Fit the pixel values to a Gaussian distribution
    mu, std = norm.fit(pixel_values)
    
    # Determine pixel value thresholds for given probabilities
    lower_thresh = norm.ppf(lower_prob, mu, std)
    upper_thresh = norm.ppf(upper_prob, mu, std)
    
    # Create a new image array with modified pixel values
    new_img = np.zeros_like(img_array)
    new_img[(img_array >= lower_thresh) & (img_array <= upper_thresh)] = 255
    
    return new_img

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert image to numpy array
    img_array = np.array(img_gray)
    
    # Apply your image processing functions
    img_filtered = apply_kernel(img_array, horizontal_kernel, vertical_kernel)
    new_img_2 = gaussian_distribution(img_array, 0.3, 1)
    
    # Perform modified AND operation on img_filtered and new_img_2
    img_combined = np.zeros_like(img_array)
    img_combined[(img_filtered == 255) & (new_img_2 == 255)] = 255
    img_combined[(img_filtered == 255) & (new_img_2 == 0)] = 100
    img_combined[(img_filtered == 0) & (new_img_2 == 255)] = 0
    
    # Write the processed frame to the output video
    out.write(cv2.cvtColor(img_combined, cv2.COLOR_GRAY2BGR))

    # Display the frame if needed
    cv2.imshow('Processed Frame', img_combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
