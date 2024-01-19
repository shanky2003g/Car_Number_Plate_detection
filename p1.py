import cv2
import pytesseract

frameWidth = 640
frameHeight = 480
nPlateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
minArea = 200
color = (255, 0, 255)

# Specify the path to your image file
image_path = "IMG-20231117-WA0004.jpg"

# Read the image
img = cv2.imread(image_path)    

if img is None:
    print(f"Failed to read the image at '{image_path}'. Exiting...")
else:
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 10)

    for (x, y, w, h) in numberPlates:
        area = w * h
        # if area > minArea:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
        imgRoi = img[y:y + h, x:x + w]
        cv2.imshow("ROI", imgRoi)

        # Perform OCR on the cropped image
        custom_config = r'--oem 3 --psm 6 outputbase digits'
        extracted_text = pytesseract.image_to_string(imgRoi, config=custom_config)
        print(f"Extracted Text: {extracted_text}")

    cv2.imshow("Result", img)
    key = cv2.waitKey(0) & 0xFF  # Wait for a key press

    if key == ord('s'):  # Save if 's' is pressed
        cv2.imwrite("p1.jpg", imgRoi)
        print("Scan Saved")

    cv2.destroyAllWindows()