import cv2
import numpy as np

def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur and edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    return image, edges

def find_document_contour(edges):
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # Looking for a quadrilateral (document shape)
            return approx

    return None


def order_points(pts):
    """ Sort points in order: top-left, top-right, bottom-right, bottom-left """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect

def warp_perspective(image, contour):
    pts = contour.reshape(4, 2)
    rect = order_points(pts)

    # Compute proper width & height
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    # Destination points for warped image
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Perspective Transform
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

    return warped

def scan_document(image_path):
    image, edges = preprocess_image(image_path)
    contour = find_document_contour(edges)

    if contour is None:
        print("No document found!")
        return None

    scanned = warp_perspective(image, contour)
    return scanned

if __name__ == "__main__":
    image_path = "sample_doc.jpg"  # Replace with actual image path
    scanned_doc = scan_document(image_path)

    if scanned_doc is not None:
        cv2.imshow("Scanned Document", scanned_doc)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
