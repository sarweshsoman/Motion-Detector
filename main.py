import threading
import winsound
import cv2
import imutils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

_, start_frame = cap.read()
start_frame = imutils.resize(start_frame, width=500)
start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
start_frame = cv2.GaussianBlur(start_frame, (21, 21), 0)

alarm = False
alarm_mode = False
alarm_counter = 0

def beep_alarm(): #This is where you can do anything that you want when the alarm is triggered
    global alarm
    for _ in range(5):
        if not alarm_mode:
            break
        print("ALARM!!!")
        winsound.Beep(2500, 1000)
    alarm = False

while True:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=500)

    if alarm_mode:
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bw = cv2.GaussianBlur(frame_bw, (5, 5), 0)

        difference = cv2.absdiff(frame_bw, start_frame)

        # Calculate the mean and standard deviation of the pixel differences
        mean_diff = difference.mean()
        std_dev_diff = difference.std()

        # Set a threshold based on the mean and standard deviation
        threshold_value = mean_diff + 2 * std_dev_diff
        _, threshold = cv2.threshold(difference, threshold_value, 255, cv2.THRESH_BINARY)

        threshold = cv2.dilate(threshold, None, iterations=2)  # Dilate to fill gaps in contours

        contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Adjust the area threshold as needed
                motion_detected = True
                break

        if motion_detected:
            alarm_counter += 1
        else:
            if alarm_counter > 0:
                alarm_counter -= 1

        cv2.imshow("Cam", threshold)


    else:
        cv2.imshow("Cam", frame)

    if alarm_counter > 20:
        if not alarm:
            alarm = True
            threading.Thread(target=beep_alarm).start()

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord(" "):
        alarm_mode = not alarm_mode
        alarm_counter = 0
    if key_pressed == ord("q"):  # Press "q" to quit
        alarm_mode = False
        break



cap.release()
cv2.destroyAllWindows()
