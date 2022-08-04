import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import sys
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):

    imshape = img.shape

    ymin_global = img.shape[0]
    ymax_global = img.shape[0]

    # left lane line variables
    all_left_grad = []
    all_left_y = []
    all_left_x = []

    # right lane line variables
    all_right_grad = []
    all_right_y = []
    all_right_x = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            gradient, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            ymin_global = min(min(y1, y2), ymin_global)

            if (gradient > 0):
                all_left_grad += [gradient]
                all_left_y += [y1, y2]
                all_left_x += [x1, x2]
            else:
                all_right_grad += [gradient]
                all_right_y += [y1, y2]
                all_right_x += [x1, x2]

    left_mean_grad = np.mean(all_left_grad)
    left_y_mean = np.mean(all_left_y)
    left_x_mean = np.mean(all_left_x)
    left_intercept = left_y_mean - (left_mean_grad * left_x_mean)

    right_mean_grad = np.mean(all_right_grad)
    right_y_mean = np.mean(all_right_y)
    right_x_mean = np.mean(all_right_x)
    right_intercept = right_y_mean - (right_mean_grad * right_x_mean)

    # Make sure we have some points in each lane line category
    if ((len(all_left_grad) > 0) and (len(all_right_grad) > 0)):
        upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
        lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
        upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
        lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)

        cv2.line(img, (upper_left_x, ymin_global),
                 (lower_left_x, ymax_global), color, thickness)
        cv2.line(img, (upper_right_x, ymin_global),
                 (lower_right_x, ymax_global), color, thickness)


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)

    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(image):
    # grayscale the image
    grayscaled = grayscale(image)

    # apply gaussian blur
    kernelSize = 5
    gaussianBlur = gaussian_blur(grayscaled, kernelSize)

    # canny
    minThreshold = 100
    maxThreshold = 200
    edgeDetectedImage = canny(gaussianBlur, minThreshold, maxThreshold)

    # apply mask
    lowerLeftPoint = [130, 540]
    upperLeftPoint = [410, 350]
    upperRightPoint = [570, 350]
    lowerRightPoint = [915, 540]

    pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint,
                    lowerRightPoint]], dtype=np.int32)
    masked_image = region_of_interest(edgeDetectedImage, pts)

    # hough lines
    rho = 1
    theta = np.pi/180
    threshold = 30
    min_line_len = 20
    max_line_gap = 20

    houged = hough_lines(masked_image, rho, theta, threshold, min_line_len,
                         max_line_gap)

    # outline the input image
    colored_image = weighted_img(houged, image)
    return colored_image


def main():

    cap = cv2.VideoCapture(
        'data/video/newdata/test_sample.mp4')  # test_sample.mp4
    # if not cap.isOpened():
    #     print('File open failed!')
    #     cap.release()
    #     sys.exit()

    ## video out ##
    # w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # delay = int(1000 / fps)

    # # create the `VideoWriter()` object
    # out = cv2.VideoWriter('result_output.mp4',
    #                       cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # img_out, angle, colorwarp, draw_poly_img = lane_finding_pipeline(
        #     frame, init, mtx, dist)

        img_out = process_image(frame)

        # Videowirte
        # out.write(img_out)

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', img_out)
        # cv2.namedWindow('colorwarp', cv2.WINDOW_NORMAL)
        # cv2.imshow('colorwarp', colorwarp)
        # cv2.namedWindow('draw_poly', cv2.WINDOW_NORMAL)
        # cv2.imshow('draw_poly', draw_poly_img)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# output2 = 'car_lane_detection2.mp4'
# clip1 = VideoFileClip("./video/solidWhiteRight.mp4")
# white_clip = clip1.fl_image(process_image)
# %time white_clip.write_videofile(output2, audio=False)
