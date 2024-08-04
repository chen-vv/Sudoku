import cv2
import numpy as np
import tensorflow as tf

def center_image(image):
    gray = image.copy()

    gray = 255 * (gray < 150).astype(np.uint8) # To invert the text to white
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    rect = image[y:y+h, x:x+w] # Crop the image - note we do this on the original image
    # cv2.imshow("Cropped", rect) # Show it
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("rect.png", rect) # Save the image

    # Thresholding
    # _, thresh = cv2.threshold(rect, 0, 255, cv2.THRESH_BINARY_INV)

    # Crop the digit using the bounding box coordinates
    digit = gray[y:y+h, x:x+w]

    # Determine center of the digit
    center_x = x + w // 2
    center_y = y + h // 2

    print(f"Center at {center_x}, {center_y}")

    # Determine dimensions for new image (make it square for simplicity)
    max_dim = max(w, h)
    new_size = max_dim + 10  # Add some padding TODO CHANGE THIS VALUE IF NEEDED

    # Create a white background image
    background = np.zeros((new_size, new_size), dtype=np.uint8)

    # Calculate offset to paste the digit in the center
    offset_x = (new_size - w) // 2
    offset_y = (new_size - h) // 2

    # Paste the digit onto the white background
    background[offset_y:offset_y+h, offset_x:offset_x+w] = digit

    white_bg_img = cv2.bitwise_not(background)

    return white_bg_img

# print("Loading model...")
# model = tf.keras.models.load_model('mnist_digit_recognition_model.keras')

# images = []

# for filename in os.listdir("extracted_cells"):
#     if os.path.isfile(os.path.join("extracted_cells", filename)):
#         images.append(filename)

# for fname in images:
#     # Load image
#     image = cv2.imread(f'extracted_cells/{fname}')

#     cv2.imshow("Original", image)
#     cv2.waitKey()

#     # Show the result (optional)
#     cv2.imshow('Centered Digit', center_image(image))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Save the result
#     # cv2.imwrite(f'{filename}.jpg', background)

# images = ["x_train_0.jpg", "x_train_1.jpg", "x_train_2.jpg"]
# model = tf.keras.models.load_model('mnist_digit_recognition_model.keras')

# for i in images:
#     img = cv2.imread(i)
#     image_batch = img.reshape((1, 28, 28, 1)) 

#     predictions = model.predict(image_batch)

#     print(predictions)
