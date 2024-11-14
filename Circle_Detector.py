import numpy as np
import cv2
import matplotlib.pyplot as plt


def ellipse_test(moments):
    size_threshold = 10
    fullfill_threshold = 0.01
    eccentricity_threshold = 0.9

    area = moments['m00']
    if(area < size_threshold):
        return False

    mx= moments['m10'] / area
    my= moments['m01']/ area
    xx = moments['mu20'] / area
    xy = moments['mu11'] / area
    yy = moments['mu02'] / area
    
    det = (xx+yy)*(xx+yy)-4*(xx*yy-xy*xy)
    if (det > 0):
        det = np.sqrt(det); 
    else:
        det = 0;
    f0 = ((xx+yy)+det)/2
    f1 = ((xx+yy)-det)/2
    m0 = np.sqrt(f0)
    m1 = np.sqrt(f1)

    ratio1 = abs(1-m1/m0)
    ratio2 = abs(1- m0*m1*4*np.pi/area)

    if(ratio2>fullfill_threshold):
        return False

    if(ratio1>eccentricity_threshold):
        return False
    return True

def cal_dist(m1, m2):
    x1 = m1['m10']/m1['m00']
    y1 = m1['m01']/m1['m00']
    x2 = m2['m10']/m2['m00']
    y2 = m2['m01']/m2['m00']
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


#preprocessing 
rgb_img =cv2.imread("../data/exp_0823/tir2/tir_004.png") # 4 7 12
filtered_img = cv2.bilateralFilter(rgb_img,-1,10,10)
gray_img = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)
blur_img = cv2.GaussianBlur(gray_img, [3,3], 0)

#extract all possible contour
bs = [3,7, 11, 15,19]
ss = [1,3,5]
ellipse_contours = []
final_contours= []
for i in range(len(bs)):
    for j in range(len(ss)):
        img_thresh = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs[i],2)  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,[ss[j],ss[j]])
        img_morph = cv2.morphologyEx(img_thresh,cv2.MORPH_CLOSE,kernel)
        contours = cv2.findContours(img_morph,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            if ellipse_test(cv2.moments(contour)):
                ellipse_contours.append(contour)


# merge same blobs 
distance_threshold = 10
for i in range(len(ellipse_contours)):
    m1 = cv2.moments(ellipse_contours[i])
    
    isnew = True
    for j in range(len(final_contours)):
        m2 = cv2.moments(final_contours[j])
        dist = cal_dist(m1,  m2)
        if (dist < distance_threshold ):
            isnew = False
            if m1['m00'] > m2['m00']:
                final_contours[j]= ellipse_contours[i]
            break

    if isnew:
        final_contours.append(ellipse_contours[i])


out_img = cv2.drawContours(rgb_img.copy(), final_contours, -1, [0, 0, 100], cv2.LINE_8)
plt.imshow(out_img)