import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

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
    sudoku_a = cv2.imread("Sudoku/9.png")
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
        
    plt.figure()
    plt.imshow(imagewrap)
    plt.show()

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

    # Extract each cell with margin adjustment
    for row in range(num_rows):
        for col in range(num_cols):
            # Define the bounding box of the cell with margin adjustment
            start_row = row * (height // num_rows) + margin_top_left
            end_row = start_row + cell_height
            start_col = col * (width // num_cols) + margin_top_left
            end_col = start_col + cell_width

            # Crop the image to get the cell
            cell = imagewrap[start_row:end_row, start_col:end_col]
            
            # Append the cell to the list
            cells.append(cell)
            
            # Optionally save the cell
            # cv2.imwrite(f'cell_{row}_{col}.jpg', cell)

            # Optionally, display the cell using matplotlib (for visualization)
            plt.imshow(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
            plt.title(f'Cell {row}_{col}')
            plt.axis('off')
            plt.show()