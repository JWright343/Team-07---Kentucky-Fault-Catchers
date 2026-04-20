#Vision system code broken down into manageable functions.

import cv2
import numpy as np

# ============================================================
# Utility Functions
# ============================================================

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image '{path}'.")
    return img


def morphology_clean(mask, kernel_size=5, op=cv2.MORPH_CLOSE):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, op, kernel)


# ============================================================
# Step 0 — Capture and save photo
# ============================================================

def capture_photo(filename="photo.jpg"):
    camera = cv2.VideoCapture(0) # 0 is the laptop camera --JCW
    if not camera.isOpened():
        raise Exception("Could not open webcam.")
        # This can also be where we put in the camera not found error code --JCW

    ret, frame = camera.read()
    camera.release()

    if not ret:
        raise Exception("Failed to capture image.")

    cv2.imwrite(filename, frame)
    print(f"✅ Photo saved as '{filename}'")

# ============================================================
# Step 1 — Green Border Mask
# ============================================================

def mask_green(hsv):
    lower = np.array([45, 110, 140]) #Lower HSV bounds
    upper = np.array([65, 255, 255]) #Upper HSV bounds, creates a range to look for
    mask = cv2.inRange(hsv, lower, upper)
    mask = morphology_clean(mask, 5, cv2.MORPH_OPEN)
    return morphology_clean(mask, 5, cv2.MORPH_CLOSE)


# ============================================================
# Step 2 — Extract ROI around device contour
# ============================================================

def extract_device_roi(image, mask_green):
    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 500]

    if not valid:
        raise Exception("No sufficiently large green region found!")

    img_center = np.array([image.shape[1] // 2, image.shape[0] // 2])

    def center_distance(c):
        M = cv2.moments(c)
        if M["m00"] == 0:
            return float("inf")
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return np.linalg.norm(np.array([cx, cy]) - img_center)

    device_contour = min(valid, key=center_distance)
    x, y, w, h = cv2.boundingRect(device_contour)

    x_end = min(x + w, image.shape[1])
    y_end = min(y + h, image.shape[0])
    roi = image[y:y_end, x:x_end]
    cv2.imshow("Region of Interest", roi)
    return roi, (x, y, x_end, y_end)


# ============================================================
# Step 3 — Mask white + red for ellipse fitting
# ============================================================

# Instead of finding white and red now, we can find non-green in the ROI,
# which should be in the green region -- JCW

def mask_white_and_red(hsv):
    # white
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 45, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # red (two ranges because red wraps hue axis)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    combined = cv2.bitwise_or(mask_white, mask_red)
    return morphology_clean(combined, 5, cv2.MORPH_CLOSE)


# ============================================================
# Step 4 — Fit ellipse to ROI mask
# ============================================================

def fit_ellipse(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Exception("No region found for ellipse fitting!")

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        raise Exception("Not enough points to fit ellipse!")

    return cv2.fitEllipse(contour)


# ============================================================
# Step 6 — Red detection inside ellipse
# ============================================================

def detect_red_inside_full_image(mask_red_roi, ellipse_mask, x, y, x_end, y_end, full_shape):
    mask_red_full = np.zeros(full_shape[:2], dtype="uint8")
    mask_red_full[y:y_end, x:x_end] = mask_red_roi

    inside = cv2.bitwise_and(mask_red_full, ellipse_mask)
    inside = morphology_clean(inside, 3, cv2.MORPH_OPEN)

    return inside


# ============================================================
# Step 7 — Compute red percentage
# ============================================================

def compute_red_percentage(red_mask, ellipse_mask):
    total = np.count_nonzero(ellipse_mask)
    red = np.count_nonzero(red_mask)
    if total == 0:
        return 0.0
    return (red / total) * 100


# ============================================================
# Main Pipeline
# ============================================================

def main():

    #while True: #This will allow the program to run indefinietely

        # Step 0: Capture photo
        capture_photo("photo.jpg") # Saves and overwrites file
    
        # Step 1: Load image + HSV
        image = load_image("photo.jpg")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Step 1: Green border mask
        green_mask = mask_green(hsv)

        # Step 2: Extract ROI
        roi, (x, y, x_end, y_end) = extract_device_roi(image, green_mask)
        roi_hsv = hsv[y:y_end, x:x_end]

        # Step 3: Mask white + red
        mask_for_ellipse = mask_white_and_red(roi_hsv)

        # Step 4: Fit ellipse
        ellipse = fit_ellipse(mask_for_ellipse)

        # Shift ellipse into global coordinates
        (cx, cy), (MA, ma), angle = ellipse
        ellipse_global = ((cx + x, cy + y), (MA, ma), angle)

        # Step 5: Create ellipse mask
        ellipse_mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipse_mask, ellipse_global, 255, -1)

        # Step 6: Red inside ellipse
        # Red mask inside ROI
        mask_red_roi = cv2.bitwise_or(
            cv2.inRange(roi_hsv, np.array([0,120,70]), np.array([10,255,255])),
            cv2.inRange(roi_hsv, np.array([170,120,70]), np.array([180,255,255]))
        )

        red_inside = detect_red_inside_full_image(mask_red_roi, ellipse_mask,
                                              x, y, x_end, y_end, image.shape)

        # Step 7: Compute coverage
        percent_red = compute_red_percentage(red_inside, ellipse_mask)

        print(f"Red coverage inside ellipse: {percent_red:.2f}%")
        if percent_red > 40:
            print("⚠️ ALERT: More than 40% red detected inside ellipse.") # FCI(s) have tripped here

            # Here would be where we signal to the indication system to operate -- JCW
        else:
            print("✅ Less than 40% red detected inside ellipse.") # No FCIs have tripped

            # If this condition occurs, then nothing should happen -- JCW

        # Step 8: Visual Debug
        debug = image.copy()
        cv2.ellipse(debug, ellipse_global, (0, 255, 0), 2)

        overlay = image.copy()
        overlay[red_inside > 0] = [0, 0, 255]

        cv2.imshow("Original w/ Ellipse", debug)
        cv2.imshow("Ellipse Mask", ellipse_mask)
        cv2.imshow("Red Inside Ellipse", red_inside)
        cv2.imshow("Overlay Red", overlay)

        cv2.imwrite("BoundingBox.jpg", debug)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
