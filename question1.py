# Import necessary libraries
import cv2  # OpenCV library for computer vision tasks
import os   # Library for interacting with the operating system

# Load the Haar cascade classifier for license plate detection
cascade_path = "assets/haarcascade_licence_plate_rus_16stages.xml"
license_plate_cascade = cv2.CascadeClassifier(cascade_path)

# Open the video file
video_path = "assets/motor.mp4"
cap = cv2.VideoCapture(video_path)

# Create output directory for saving detected license plates if it doesn't exist
output_dir = "./license_plates"
os.makedirs(output_dir, exist_ok=True)

# Function to save the detected license plate image
def save_license_plate(frame, x, y, w, h, count):
    # Crop the license plate region from the frame
    plate_img = frame[y:y+h, x:x+w]
    # Define the filename and path to save the license plate image
    filename = os.path.join(output_dir, f"license_plate_{count}.jpg")
    # Save the cropped license plate image to the specified file
    cv2.imwrite(filename, plate_img)
    print(f"License plate saved to {filename}")

# Initialize a counter for naming saved license plate images
count = 0

# Create a window to display the video frames
cv2.namedWindow('License Plate Detection', cv2.WINDOW_NORMAL)

# Main loop to read and process frames from the video
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:  # If no frame is read (end of video), break the loop
        break

    # Convert the frame to grayscale as the cascade classifier works on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the grayscale frame
    license_plates = license_plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))

    # Draw rectangles around detected license plates
    for (x, y, w, h) in license_plates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue rectangle with thickness of 2

    # Display the frame with detected license plates
    cv2.imshow('License Plate Detection', frame)

    # Wait for the user to press a key
    key = cv2.waitKey(1)
    if key == ord('s'):  # If 's' key is pressed, save the detected license plates
        for (x, y, w, h) in license_plates:
            save_license_plate(frame, x, y, w, h, count)  # Save each detected license plate
            count += 1  # Increment the counter for the next license plate
    elif key == ord('q'):  # If 'q' key is pressed, quit the program
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
