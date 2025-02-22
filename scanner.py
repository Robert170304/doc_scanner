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

def warp_perspective(image, contour):
    pts = contour.reshape(4, 2)
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Compute new dimensions
    (tl, tr, br, bl) = rect
    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # Apply perspective transformation
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (int(width), int(height)))

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
