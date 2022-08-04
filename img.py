import cv2
vidcap = cv2.VideoCapture('/data/video/newdata/test_sample.mp4')
success, image = vidcap.read()
count = 0
while success:
    image = cv2.resize(image, (1080, 720))
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
