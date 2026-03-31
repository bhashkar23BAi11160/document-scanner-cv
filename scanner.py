import cv2
import numpy as np

# Load image
image = cv2.imread("input.jpg")
orig = image.copy()

# Resize for easier processing
image = cv2.resize(image, (500, 700))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blur, 30, 200)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

doc = None

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        doc = approx
        break

# Draw contour
cv2.drawContours(image, [doc], -1, (0, 255, 0), 2)

# Perspective transform
pts = doc.reshape(4, 2)
rect = np.zeros((4, 2), dtype="float32")

s = pts.sum(axis=1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]

diff = np.diff(pts, axis=1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]

(tl, tr, br, bl) = rect

widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
maxWidth = max(int(widthA), int(widthB))

heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
maxHeight = max(int(heightA), int(heightB))

dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]
], dtype="float32")

M = cv2.getPerspectiveTransform(rect, dst)
warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

# Convert to grayscale for scan effect
gray_warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

# Apply threshold (this is the magic)
_, scan = cv2.threshold(gray_warp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Save output
cv2.imwrite("output/scanned.jpg", scan)

# Show images
cv2.imshow("Original", orig)
cv2.imshow("Edges", edges)
cv2.imshow("Scanned", scan)

cv2.waitKey(0)
cv2.destroyAllWindows()