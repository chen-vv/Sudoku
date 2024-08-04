import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from test import center_image

### Helper Functions ###

"""
Converts the image to grayscale then applies Gaussian blur
and adaptive threshold to it.
"""
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (3,3),6) 
    threshold_img = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    return threshold_img

"""
This function returns a set of (x, y) points and the largest
contour area.
"""
def main_outline(contour):
    biggest = np.array([])
    max_area = 0
    for i in contour:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i , 0.02* peri, True)
            if area > max_area and len(approx) ==4:
                biggest = approx
                max_area = area
    return biggest,max_area

def reframe(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4,1,2),dtype = np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis =1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new

def splitcells(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes


if __name__ == "__main__":
    sudoku_a = cv2.imread("Sudoku/10.png")
    sudoku_a = cv2.resize(sudoku_a, (450,450))

    threshold = preprocess(sudoku_a)

    # Finding the outline of the sudoku puzzle in the image
    contour_1 = sudoku_a.copy()
    contour_2 = sudoku_a.copy()
    contour, hierarchy = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_1, contour,-1,(0,255,0),3)

    black_img = np.zeros((450,450,3), np.uint8)
    biggest, maxArea = main_outline(contour)
    if biggest.size != 0:
        biggest = reframe(biggest)
        cv2.drawContours(contour_2,biggest,-1, (0,255,0),10)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
        matrix = cv2.getPerspectiveTransform(pts1,pts2)  
        imagewrap = cv2.warpPerspective(sudoku_a,matrix,(450,450))
        imagewrap =cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)
        
    # plt.figure()
    # plt.imshow(imagewrap)
    # plt.show()

    height, width = imagewrap.shape
    print(f"Height:{height}")
    print(f"Width: {width}")

    # Define grid size
    num_rows = 9
    num_cols = 9

    # Change this margin value to control how inside the cell should
    # be cropped
    margin = 10

    # Calculate the size of each cell, with margin adjustment
    cell_height = (height // num_rows) - margin
    cell_width = (width // num_cols) - margin

    # Calculate margin offsets
    margin_top_left = margin // 2
    margin_bottom_right = margin - margin_top_left

    # Create a list to store cell images
    cells = []

    print("Loading model...")
    model = tf.keras.models.load_model('mnist_digit_recognition_model.keras')

    # Extract each cell with margin adjustment
    for row in range(num_rows):
        for col in range(num_cols):
            # Define the bounding box of the cell with margin adjustment
            # to remove grid lines that may appear on the edges
            start_row = row * (height // num_rows) + margin_top_left
            end_row = start_row + cell_height
            start_col = col * (width // num_cols) + margin_top_left
            end_col = start_col + cell_width

            # Crop the image to get the cell
            old_cell = imagewrap[start_row:end_row, start_col:end_col]
            cell = center_image(old_cell)

            # Resize the cell to MNIST size
            resized_cell = cv2.resize(cell, (28,28), interpolation=cv2.INTER_AREA)

            # Normalize the image
            normalized_cell = cv2.normalize(resized_cell, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            image_batch = normalized_cell.reshape((1, 28, 28, 1)) 

            # Optionally, display the cell using matplotlib (for visualization)
            plt.imshow(cv2.cvtColor(resized_cell, cv2.COLOR_BGR2RGB))
            plt.title(f'Cell {row}_{col}')
            plt.axis('off')
            plt.show()

            # Predict the digit
            predictions = model.predict(image_batch)
            predicted_digit = np.argmax(predictions[0])

            print(f'Predicted digit: {predicted_digit}')