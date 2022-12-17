import numpy as np
import cv2
import os
import math

def clear_directory(path):

    path = os.path.abspath(path)
    #Clear Output Folder
    directoryToClean = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    directoryItems = [os.path.join(directoryToClean, f) for f in os.listdir(directoryToClean)]
    [os.remove(f) for f in directoryItems if os.path.isfile(f)]
    print("[INFO] Directory cleared: ", path)

def draw_edges(path, lines):

    path = os.path.abspath(path)
    gray = cv2.imread('D:\sudoku\sudoku.jpg')

    a = len(lines)
    for i in range(a):
        img_copy = gray.copy()
        cv2.line(img_copy, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
        img_path = os.path.join(path,f'line{i}.jpg')
        cv2.imwrite(img_path,img_copy)  



def segregate_vertical_horizontal_lines(lines):

    vertical =[]
    horizontal = []

    a,_,_ = lines.shape
    for i in range(a):
        if (lines[i][0][0] - lines[i][0][1] < 0) and (lines[i][0][0] < lines[i][0][2]):
        #if (lines[i][0][2] - lines[i][0][1])/ (lines[i][0][0] - lines[i][0][0]):
            horizontal.append(lines[i])
        else:
            vertical.append(lines[i])
    return vertical,horizontal


def filter_edges(lines, line_type):

    filtered_edges = []
    distance = []
    coordinates = []

    if line_type == "vertical":
        for i in range(len(lines)):
            coordinates.append(lines[i][0][0])

    else:
        for i in range(len(lines)):
            coordinates.append(lines[i][0][1])
    
    for i in range(len(coordinates)-1):
        distance.append(coordinates[i+1] - coordinates[i])

    avg_distance = (max(distance) + min(distance))/2
    for i in range(len(coordinates) -1):
        if distance[i] > avg_distance:
            filtered_edges.append(lines[i])
    filtered_edges.append(lines[len(lines)-1])

    return filtered_edges


# # Clear Output Folder
# clear_directory('edges_output/all_edges')
# clear_directory('edges_output/segregated_edges/vertical')
# clear_directory('edges_output/segregated_edges/horizontal')
# clear_directory('edges_output/filtered_edges/vertical')
# clear_directory('edges_output/filtered_edges/horizontal')


# #Pre-process Image
# gray = cv2.imread('D:\sudoku\sudoku.jpg') #Convert to greay scale
# edges = cv2.Canny(gray,50,150,apertureSize = 3) #Get edges
# cv2.imwrite('edges-50-150.jpg',edges) 

# #Get edges - HoughLinesP
# w,h = edges.shape
# minLineLength= max(math.floor(w-(w*0.10)),math.floor(h-(h*0.10)))
# lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/360, threshold=150,lines=np.array([]), minLineLength=minLineLength,maxLineGap=100)
# draw_edges('edges_output/all_edges/',lines)
# print("Number of detected lines:",len(lines))

# #Get vertical & horizontal lines:
# vertical, horizontal = segregate_vertical_horizontal_lines(lines)
# print("Number of vertical lines:",len(vertical))
# print("Number of horizontal lines:",len(horizontal))

# #Sort Edges:
# vertical_sorted = sorted(vertical, key=lambda i: i[0][0])
# horizontal_sorted = sorted(horizontal, key=lambda i: i[0][1])
# draw_edges('edges_output/segregated_edges/vertical',vertical_sorted)
# draw_edges('edges_output/segregated_edges/horizontal',horizontal_sorted)

# # Filter excess lines based on - Distance between lines
# vertical_filtered = filter_edges(vertical_sorted,"vertical")
# draw_edges('edges_output/filtered_edges/vertical',vertical_filtered)
# horizontal_filtered = filter_edges(horizontal_sorted,"horizontal")
# draw_edges('edges_output/filtered_edges/horizontal',horizontal_filtered)

