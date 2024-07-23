import cv2  # OpenCV library for computer vision tasks
from pyzbar.pyzbar import decode  # Pyzbar library for decoding QR codes
import matplotlib.pyplot as plt  # Matplotlib library for displaying images

# Load and display the image containing QR codes
image = cv2.imread('assets/QRcode.PNG')  # Read the image from the specified file
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert the image from BGR (OpenCV format) to RGB and display it
plt.axis('off')  # Turn off axis labels and ticks
plt.show()  # Show the image

# Convert the image to grayscale for better QR code detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Decode QR codes in the grayscale image
qr_codes = decode(gray)

# Iterate over detected QR codes and write data on the image
for qr_code in qr_codes:
    (x, y, w, h) = qr_code.rect  # Get the bounding box coordinates of the QR code
    # Draw a green rectangle around the QR code
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Write the decoded QR code data above the rectangle in red color
    cv2.putText(image, qr_code.data.decode("utf-8"), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the image with the QR code data annotated
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert the image from BGR to RGB and display it
plt.axis('off')  # Turn off axis labels and ticks
plt.show()  # Show the annotated image
