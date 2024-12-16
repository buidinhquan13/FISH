import cv2
import numpy as np

def create_object():
    # dimension image
    width, height = 512,512

    # Tạo ảnh nền trắng
    image = np.ones((height, width, 1), dtype=np.uint8) * 0

    # Tọa độ các đỉnh của tam giác
    points = np.array([
        [50, 100],
        [100, 400], # Đỉnh trên
        [400, 400],   # Đỉnh trái
        [400, 150],
           # Đỉnh phải
    ], np.int32)

    # Vẽ hình tam giác
    points = points.reshape((-1, 1, 2))  # Định dạng cho OpenCV
    cv2.polylines(image, [points], isClosed=True, color=(255), thickness=1) 
    cv2.fillPoly(image, [points], color=(255))

    return image

def create_grid():
    # Kích thước ảnh
    width, height = 512, 512

    # Tạo ảnh nền trắng
    image = np.ones((height, width, 1), dtype=np.uint8) * 255

    # Kích thước ô lưới và kích thước lỗ vuông
    grid_size = 100  # Kích thước mỗi ô lưới
    hole_size = 98  # Kích thước mỗi lỗ hình vuông
    arr_cut = []# mảng từng viên
    
    num_cols = 0

    # Vẽ lưới với các lỗ
    for y in range(0, height, grid_size):

        for x in range(0, width, grid_size):
            # Tính tọa độ tâm của mỗi ô lưới
            center_x, center_y = x + grid_size // 2, y + grid_size // 2
            
            # Tính tọa độ góc của lỗ hình vuông
            top_left = (center_x - hole_size // 2, center_y - hole_size // 2)
            bottom_right = (center_x + hole_size // 2, center_y + hole_size // 2)

            top_left_v = (center_x - grid_size // 2, center_y - grid_size // 2)
            bottom_right_v = (center_x + grid_size // 2, center_y + grid_size // 2)
            
            
            # Vẽ lỗ hình vuông (tô màu đen)
            cv2.rectangle(image, top_left, bottom_right, (0), -1)
            
            if(bottom_right_v[0] > width or bottom_right_v[1] > width):
                continue
            arr_cut.append(([top_left_v, bottom_right_v]))
            if y == 0:
                num_cols += 1
    
    arr_2_cut = np.zeros((int(len(arr_cut)/num_cols),num_cols))
    #print(type(arr_cut))
    arr_cut = np.array(arr_cut)
    #print(arr_cut.shape)
    #arr_cut = arr_cut.reshape((5,5))
    return image, arr_cut, arr_2_cut

def cv_sq(sq):
    cv2.imwrite('sample.png',sq)
    contours, _ = cv2.findContours(sq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        perimeter = cv2.arcLength(contour, True)  # True indicates a closed contour
        # print(f"Object {i + 1} Perimeter: {perimeter}")

        # # Optionally, draw the contours for visualization
        # binary = cv2.drawContours(sq, [contour], -1, (127), 2)
        # cv2.imshow("Contours", binary)
        # cv2.waitKey(0)
        return perimeter
    
def cv_sq_inv(sq):
    sq = 255 - sq
    cv2.imwrite('sample.png',sq)
    contours, _ = cv2.findContours(sq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
            perimeter = cv2.arcLength(contour, True)  # True indicates a closed contour

            return perimeter


width, height = 512, 512
img_object = create_object()
#print(img_object.shape)
img_grid, arr_cut, arr_2_cut = create_grid()
#print(len(arr_cut))
# kết hợp hai ảnh object và grid
img_combine = img_object + img_grid


#tách object
img_cut = img_combine.copy()
#print(type(img_cut))
img_cut[img_grid == 255] = 0


#temp = img_cut[arr_cut[0][0][0]:arr_cut[0][1][0],arr_cut[0][0][1]:arr_cut[0][1][1]]


# print(arr_2_cut.shape)
area_arr = arr_2_cut.copy()
peri_arr = arr_2_cut.copy()
peri_arr_inv = arr_2_cut.copy()

i = 0
j = 0
d = 2
count = 0
for sq in arr_cut:
    count += 1
    # print(sq)
    area_i = np.sum(img_cut[sq[0][0]+2:sq[1][0]-2,sq[0][1]+2:sq[1][1]-2])   

    if (area_i >   94*94*255-255*300 or area_i == 0):
        
        img_cut[sq[0][0]+2:sq[1][0]-2,sq[0][1]+2:sq[1][1]-2] = 0
        
        i += 1
        if i == 5:
            j+=1
            i = 0
        
        continue
 
    #print(count)
    

    area_arr[i][j] = area_i

    
    cv_i = cv_sq(img_cut[sq[0][0]+2:sq[1][0]-2,sq[0][1]+2:sq[1][1]-2])
    cv_i_inv = cv_sq_inv(img_cut[sq[0][0]+2:sq[1][0]-2,sq[0][1]+2:sq[1][1]-2])

    peri_arr[i][j] = cv_i
    peri_arr_inv[i][j] = cv_i_inv




    i += 1
    if i == 5:
        j+=1
        i = 0
    

print('-------------------')
print(area_arr)
print('-------------------')
print(peri_arr)
print('-------------------')
print(peri_arr_inv)
S = area_arr
P1 = peri_arr
P2 = peri_arr_inv


target_area = 2550000
threshold_area = 0.15
threshold_perimeter = 0.05

used = set()

result = []
for i1 in range(S.shape[0]):
    for j1 in range(S.shape[1]):
        for i2 in range(S.shape[0]):
            for j2 in range(S.shape[1]):
                
                if (i1 == i2 and j1 == j2):
                    continue
                # Bỏ qua các ô không có giá trị
                if S[i1, j1] == 0 or S[i2, j2] == 0:
                    continue
                
                if (i1, j1) in used or (i2, j2) in used:
                    continue
                # Kiểm tra điều kiện diện tích
                area_sum = S[i1, j1] + S[i2, j2]
                if not (target_area * (1 - threshold_area) <= area_sum <= target_area):
                    continue

                # Kiểm tra điều kiện chu vi
                if abs(P1[i1, j1] - P2[i2, j2]) > threshold_perimeter * P1[i1, j1]:
                    continue

                # Lưu cặp thỏa mãn
                result.append(((i1, j1), (i2, j2), area_sum, abs(P1[i1, j1] - P2[i2, j2])))
                used.add((i1, j1))
                used.add((i2, j2))

# Hiển thị kết quả
for pair in result:
    print(f"Cặp ô lưới: {pair[0]} và {pair[1]}, Diện tích tổng: {pair[2]:.2f}, Chênh lệch chu vi: {pair[3]:.2f}")
    






# #print(arr_cut)
# cv2.imshow('grid',img_grid)
# cv2.imshow('object',img_object)
cv2.imshow('Image',img_cut)
cv2.waitKey(0)
cv2.destroyAllWindows()