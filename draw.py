import cv2
import numpy as np
drawing = False  # True if mouse is pressed
ix, iy = -1, -1  # starting position coordinates
image, mask = None, None


def draw(ref_image,size):


    mask = np.zeros((size,size,3))
    # image = np.array(ref_image,dtype=np.uint8)
    image=cv2.imread(ref_image)
    image=cv2.resize(image,(size,size))


    def draw_circle(event, x, y, flags, param):
        global ix, iy, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(image, (x, y), 15, (0, 0, 255), -1)
                cv2.circle(mask, (x, y), 15, (255, 255, 255), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(image, (x, y), 15, (0, 0, 255), -1)
            cv2.circle(mask, (x, y), 15, (255, 255, 255), -1)

    cv2.namedWindow('draw')
    cv2.setMouseCallback('draw', draw_circle)

    while True:
        # break
        cv2.imshow('draw', image)
        # cv2.imshow('im', mask)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:  # Press ESC to exit
            break
    cv2.destroyAllWindows()
    mask=cv2.GaussianBlur(mask, (9, 9), 0)/255
    return mask