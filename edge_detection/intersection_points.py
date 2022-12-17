import cv2
import math
from math import trunc
import numpy as np
import os


import detect_edges

def line_intersection(line1,line2):
    
    line1 = line1[0].tolist()
    line2 = line2[0]
    
    line1 = [[line1[0], line1[1]],[line1[2], line1[3]]]
    line2 = [[line2[0], line2[1]],[line2[2], line2[3]]]

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a,b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('Lines do not interesect')
    
    d = (det(*line1), det(*line2))
    x = trunc(det(d, xdiff) / div)
    y = trunc(det(d, ydiff) / div)

    return (x,y)

# Clear Output Folder
detect_edges.clear_directory('edges_output/all_edges')
detect_edges.clear_directory('edges_output/segregated_edges/vertical')
detect_edges.clear_directory('edges_output/segregated_edges/horizontal')
detect_edges.clear_directory('edges_output/filtered_edges/vertical')
detect_edges.clear_directory('edges_output/filtered_edges/horizontal')
detect_edges.clear_directory('squares/')

#Pre-process Image
gray = cv2.imread('D:\sudoku\sudoku.jpg') #Convert to greay scale
orginal_img = cv2.imread('D:\sudoku\sudoku.jpg')
edges = cv2.Canny(gray,50,150,apertureSize = 3) #Get edges
cv2.imwrite('edges-50-150.jpg',edges) 

#Get edges - HoughLinesP
w,h = edges.shape
minLineLength= max(math.floor(w-(w*0.10)),math.floor(h-(h*0.10)))
lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/360, threshold=150,lines=np.array([]), minLineLength=minLineLength,maxLineGap=100)
detect_edges.draw_edges('edges_output/all_edges/',lines)
print("Number of detected lines:",len(lines))

#Get vertical & horizontal lines:
vertical, horizontal = detect_edges.segregate_vertical_horizontal_lines(lines)
print("Number of vertical lines:",len(vertical))
print("Number of horizontal lines:",len(horizontal))


#Sort Edges:
vertical_sorted = sorted(vertical, key=lambda i: i[0][0])
horizontal_sorted = sorted(horizontal, key=lambda i: i[0][1])
detect_edges.draw_edges('edges_output/segregated_edges/vertical',vertical_sorted)
detect_edges.draw_edges('edges_output/segregated_edges/horizontal',horizontal_sorted)

# Filter excess lines based on - Distance between lines
vertical_filtered = detect_edges.filter_edges(vertical_sorted,"vertical")
detect_edges.draw_edges('edges_output/filtered_edges/vertical',vertical_filtered)
horizontal_filtered = detect_edges.filter_edges(horizontal_sorted,"horizontal")
detect_edges.draw_edges('edges_output/filtered_edges/horizontal',horizontal_filtered)

#----------------------------------------------------------------
#           Intersection Points
#----------------------------------------------------------------

intersection_points= []
intersection_index = 0
for hline in horizontal_filtered:

    hline_points = []

    for vline in vertical_filtered:

        img_copy = orginal_img.copy()
        center_coordinates = line_intersection(hline, vline)
        #print(f'Vertical Line: {vline}  Horizontal Line: {hline}')
        #print(f'Intersection:{center_coordinates} \n\n')
        hline_points.append(center_coordinates)
        cv2.circle(img_copy, center_coordinates, 10, (255, 255, 0), 5)
        cv2.line(img_copy, (vline[0][0], vline[0][1]), (vline[0][2], vline[0][3]), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(img_copy, (hline[0][0], hline[0][1]), (hline[0][2], hline[0][3]), (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imwrite(f"intersections/{intersection_index}.jpg", img_copy)
        intersection_index = intersection_index + 1

    
    intersection_points.append(hline_points)
print(intersection_points)

#----------------------------------------------------------------
#           Cropping squares
#----------------------------------------------------------------

square_number = 0
for i in range(0,8):
     #rows
    for j in range(0,7): #colums

        point1 = intersection_points[i][j] #top left vertex
        point2 = intersection_points[i+1][j+1] #bottom right vertex

        print(point1)
        print(point2)


        img_copy = orginal_img[point1[1]: point2[1], point1[0]: point2[0]].copy()
        
        # cv2.circle(img_copy, point1, 10, (255, 255, 0), 5)
        # cv2.circle(img_copy, point2, 10, (255, 255, 0), 5)

        cv2.imwrite(f"squares/square{square_number}.jpg", img_copy)
        square_number = square_number +1 






