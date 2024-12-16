import cv2
import numpy as np
from ultralytics import YOLO
import random

# Load the model
model = YOLO("yolo11n-seg.pt")
results = model("image.jpg", save=False)
image = results[0].plot(labels=False)

# Size of each grid square in pixels
square_size_px = 100
height, width, _ = image.shape
significant_cells = []

# Kiểm tra chênh lệch k quá 10% 
# Có thể thay đổi threshold 
def is_area_within_range(area1, area2, target_area, threshold=3):
    total_area = area1 + area2
    difference_percentage = abs(total_area - target_area) / target_area * 100
    return difference_percentage <= threshold

# Extract segmentation masks from the results
for detection in results[0].boxes:
    
    x1, y1, x2, y2 = detection.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cell_counter = 1

    # Loop through each grid cell inside the bounding box
    for y in range(y1, y2, square_size_px):
        for x in range(x1, x2, square_size_px):
            # Create a binary mask for the current grid cell
            grid_cell_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(grid_cell_mask, (x, y), (x + square_size_px, y + square_size_px), 255, -1)

            # For each mask (segmentation), calculate the intersection with the grid cell
            total_area = 0
            for mask in results[0].masks.xy:
                mask = np.array(mask, dtype=np.int32)
                segmented_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(segmented_mask, [mask], 255)

                # Calculate the intersection between the grid cell mask and the segmented mask
                intersection = cv2.bitwise_and(grid_cell_mask, segmented_mask)
                intersection_area = np.sum(intersection > 0)  # Area of the intersection
                total_area += intersection_area
                
                perimeter = 0
                for i in range(y, min(y + square_size_px, height)):  # Duyệt theo hàng
                    for j in range(x, min(x + square_size_px, width)):  # Duyệt theo cột
                        if intersection[i, j] == 255:  # Pixel thuộc phần giao
                            # Kiểm tra pixel biên
                            if (i > 0 and intersection[i-1, j] == 0) or \
                               (i < height - 1 and intersection[i+1, j] == 0) or \
                               (j > 0 and intersection[i, j-1] == 0) or \
                               (j < width - 1 and intersection[i, j+1] == 0):
                                perimeter += 1
                
                # dilated_mask = cv2.dilate(segmented_mask, np.ones((3, 3), np.uint8), iterations=1)
                # perimeter_mask = dilated_mask - segmented_mask  # Các pixel biên là sự khác biệt

                # # Tính chu vi bằng cách đếm các pixel biên có giá trị 1
                # perimeter = np.sum(perimeter_mask == 255) 
                
                # perimeter = 0
                # for i in range(x , min(x + square_size_px, width - 1)):  # tránh các pixel biên của ảnh
                #     for j in range(y , min(y + square_size_px, height - 1)):
                #         if segmented_mask[i, j] == 255:  # pixel có giá trị 255 (của miếng cá)
                #             # Kiểm tra xem có ít nhất một pixel lân cận có giá trị 0 không
                #             if (segmented_mask[i-1, j] == 0 or segmented_mask[i+1, j] == 0 or
                #                 segmented_mask[i, j-1] == 0 or segmented_mask[i, j+1] == 0):
                #                 perimeter += 1
            
            ######### Mới thêm
            if total_area > square_size_px * square_size_px:
                total_area = square_size_px * square_size_px
            difference_percentage = is_area_within_range(total_area, 0, square_size_px * square_size_px)

            # Create a semi-transparent overlay for the grid cell
            overlay = image.copy()
            
            if difference_percentage:
                color = (0, 255, 0)  
                overlay = image.copy()
                cv2.rectangle(overlay, (x, y), (x + square_size_px, y + square_size_px), color, -1)
                cv2.addWeighted(overlay, 0.5, image, 0.5, 2, image) # làm mờ
            else:
                significant_cells.append(((x, y), total_area,perimeter))
                
            cell_center_x = x + 5
            cell_center_y = y + 20

            # Place the area value at the top-left corner of each cell
            cv2.putText(image, f'{total_area}px', (cell_center_x, cell_center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

            # Increment the cell counter
            cell_counter += 1
                   
significant_cells = sorted(significant_cells, key=lambda x: x[1], reverse=True)             
print(significant_cells)
groups = []
check = []

for i in range(len(significant_cells)):   
    if i in check:
        continue
    (x1, y1), area1, p1 = significant_cells[i]
    #for j in range(i+1, len(significant_cells)):
    for j in range(len(significant_cells) - 1, i, -1):
        if j in check:
            continue
        
        #(x1, y1), area1 = significant_cells[i]
        (x2, y2), area2, p2 = significant_cells[j]

        if is_area_within_range(area1, area2, square_size_px * square_size_px):
            added_to_group = False
            for group in groups:
                if (i in group or j in group):
                    group.add(i)
                    group.add(j)
                    added_to_group = True
                    break
            
            if not added_to_group:
                # Create a new group for this pair
                groups.append({i, j})
            check.append(i)
            check.append(j)
            
            break

# Iterate over the groups and draw the cells
for group_index, group in enumerate(groups, start=1):
    # Random color for the group
    random_color = [random.randint(0, 255) for _ in range(3)]

    # Iterate over all cells in this group and draw them with the same color
    for i in group:
        (x, y), area, p = significant_cells[i]

        # Draw the cell with the random color and 30% opacity
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + square_size_px, y + square_size_px), random_color, -1)
        cv2.addWeighted(overlay, 0.3, image, 1 - 0.3, 0, image)

        # Display the group number inside the grid cell
        text_position = (x + square_size_px // 4, y + square_size_px // 2)
        cv2.putText(image, str(group_index), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

cv2.imwrite("result.jpg", image)
cv2.imshow("YOLO Result with Semi-Transparent Grid and Area", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
