from lib.device import Camera
from lib.processors_noopenmdao import findFaceGetPulse
from lib.interface import plotXY, imshow, waitKey, destroyWindow
from cv2 import moveWindow
import argparse  ##cli
import numpy as np
import datetime
import socket  ###
import sys  ###

import cv2
import face_recognition
import csv
from twilio.rest import Client
import pandas as pd


class getPulseApp(object):
    # default constructor
    def __init__(self, args):

        self.cameras = []
        self.selected_cam = 0
        for i in range(3):
            camera = Camera()  # first camera by default...class in device.py
            if camera.valid or not len(self.cameras):
                self.cameras.append(camera)
            else:
                break
        self.w, self.h = 0, 0
        self.pressed = 0  ##maybe the setting
        ###################changed bpm limits
        self.processor = findFaceGetPulse(
            bpm_limits=[60, 100], data_spike_limit=2500.0, face_detector_smoothness=10.0
        )  ##touple cant be modified?
        self.bpm_plot = False
        # self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

        self.key_controls = {
            "s": self.toggle_search,
            "d": self.toggle_display_plot,
            "f": self.write_csv,
        }  ##dictionary

    def write_csv(self):
        """
        Writes current data to a csv file
        """
        # fn = "Pulse " + str(datetime.datetime.now())  ##file name when 'f'
        fn = "Pulse " + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        fn = fn.replace(":", "_").replace(".", "_")  ##replace the : .
        data = np.vstack(
            (self.processor.times, self.processor.samples)
        ).T  ##TRANSPOSE THE DATA......................is this understooddd?
        np.savetxt(fn + ".csv", data, delimiter=",")
        print("Writing csv")

    def toggle_search(self):
        """
        Toggles a motion lock on the processor's face detection component.

        Locking the forehead location in place significantly improves
        data quality, once a forehead has been sucessfully isolated.
        """
        # state = self.processor.find_faces.toggle()
        state = self.processor.find_faces_toggle()
        print("face detection lock =", not state)

    def toggle_display_plot(self):
        """
        Toggles the data display.
        """
        if self.bpm_plot:
            print("bpm plot disabled")
            self.bpm_plot = False
            destroyWindow(self.plot_title)
        else:
            print("bpm plot enabled")
            if self.processor.find_faces:
                self.toggle_search()
            self.bpm_plot = True
            self.make_bpm_plot()
            moveWindow(self.plot_title, self.w, 0)

    def make_bpm_plot(self):
        """
        Creates and/or updates the data display
        """
        plotXY(
            [
                [self.processor.times, self.processor.samples],
                [self.processor.freqs, self.processor.fft],
            ],
            labels=[False, True],
            showmax=[False, "bpm"],
            label_ndigits=[0, 0],
            showmax_digits=[0, 1],
            skip=[3, 3],
            name=self.plot_title,
            bg=self.processor.slices[0],
        )

    def key_handler(self):
        """
        Handle keystrokes, as set at the bottom of __init__()

        A plotting or camera frame window must have focus for keypresses to be
        detected.
        """

        self.pressed = waitKey(10) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("Exiting")
            for cam in self.cameras:
                cam.cam.release()
                cv2.destroyAllWindows()
            sys.exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    def main_loop(self, writ):
        """
        Single iteration of the application's main loop.
        """
        # Get current image frame from the camera
        frame = self.cameras[self.selected_cam].get_frame()
        
        self.h, self.w, _c = frame.shape

        # display unaltered frame
        # imshow("Original",frame)

        # set current image frame to the processor's input
        self.processor.frame_in = frame
        # process the image frame to perform all needed analysis
        self.processor.run(self.selected_cam)
        # collect the output frame for display
        output_frame = self.processor.frame_out

        # show the processed/annotated output frame
        imshow("Processed", output_frame)

        # create and/or update the raw data display if needed
        if self.bpm_plot:
            self.make_bpm_plot()

        variable = []
        counter = 0
        my_current = 0

        ############################################
        if self.processor.bpm != 0:
            my_current = str(self.processor.bpm)

            # print(my_current)  # PRINT THE PULSE   ONTO SACREEN

            writ.writerow(
                (
                    str(self.processor.bpm),
                    str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                )
            )  ##work with writerows also but will come as individual characters

            # print("Writing csv")
            print(my_current)

        # handle any key presses
        self.key_handler()


if __name__ == "__main__":

    #####my_face_recog
    print("\n Inside face-recognition now")
    # object=fr()
    # object.function()

    # This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
    # other example, but it includes some basic performance tweaks to make things run a lot faster:
    #   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
    #   2. Only detect faces in every other frame of video.

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Cannot open the camera")
        exit()
    
    # Load a sample picture and learn how to recognize it.
    obama_image = face_recognition.load_image_file("obama.jpg")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    shekhar_image = face_recognition.load_image_file("shekhar.jpg")
    shekhar_face_encoding = face_recognition.face_encodings(shekhar_image)[0]

    pursh_image = face_recognition.load_image_file("pursh.jpg")
    pursh_face_encoding = face_recognition.face_encodings(pursh_image)[0]

    # Load a second sample picture and learn how to recognize it.
    mushtaq_image = face_recognition.load_image_file("mushtaq.jpg")
    mushtaq_face_encoding = face_recognition.face_encodings(mushtaq_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        obama_face_encoding,
        mushtaq_face_encoding,
        shekhar_face_encoding,
        pursh_face_encoding,
    ]
    known_face_names = ["Barack Obama", "mushtaq", "shekhar", "purshottam"]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    my_success = False
    my_count = 0
    my_facenames = []

    while my_count <= 100:
        # Grab a single frame of video i.e capture frome-by-frame
        ret, frame = video_capture.read()

        if not ret:
            print("Can't receive a frame. Exiting ...")
            break
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations
            )

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding
                )
                name = "Unknown"

                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                face_names.append(name)
                # my_facenames=face_names[:]
                print(face_names)
                if face_names[0] == "Unknown":
                    exit(0)
                    # video_capture.release()
                    # cv2.destroyAllWindows()
                # print (my_facenames)
        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
            )

        # Display the resulting image
        cv2.imshow("Video", frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        my_count = my_count + 1

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

    print("\n Inside pulse module now")
    COUNT = 500
    parser = argparse.ArgumentParser(description="Webcam pulse detector.")
    parser.add_argument(
        "--serial", default=None, help="serial port destination for bpm data"
    )

    args = parser.parse_args()
    App = getPulseApp(args)
    ##
    fn = "Pulse " + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    fn = fn + ".csv"
    var1 = open(fn, "w+")  ##need to append
    #    var = open('test.csv', 'w+')
    writer = csv.writer(var1)
    writer.writerow(("pulse", "timestamp"))

    while COUNT != 0:
        App.main_loop(writer)

        COUNT = COUNT - 1

    print("Done with pulse")

    #    data = pd.read_csv('test.csv')
    data = pd.read_csv(fn)
    pulses = np.array(data["pulse"])
    # print(pulses)###########yours
    #  x=np.array(pulses)
    #  print(x)
    # print(" are:",pulses[pulses>65])

    ##################################################You commented this one
    user_value = pulses.mean()
    ###wromg print("overall average: %.2f"% user_value)

    big = pulses[pulses > 65]  ###I DID
    # big=pulses[pulses<120]
    value = big.mean()
    # print(big)
    ###print("\ntest Pulse of user is: %.2f"% value)
    maxsize = 50
    realtime = big[-maxsize:]
    #######print(realtime)
    # answer = %.2f % realtime.maen()####################
    print("pulse is: %.2f" % realtime.mean())
    # print("average pulse of user is:",pulses.mean())
    # user_value=pulses.mean()
    # print(user_value)

    # fd = open('everyday.csv','a')
    # wri = csv.writer(fd)
    # wri.writerow('average/day')
    # wri.writerow((str(user_value)))
    with open("some.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(str(user_value))

    # twilio implementation

    account_sid = "AC85af95b3fc0c19a9389e8e6a56dbbee8"
    auth_token = "5e1c31e3c57f08dfd2feb445485af4f6"
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        "+818025650498",
        body="Your kin's pulse rate is !" + str(user_value),
        from_="+15512223358",
        # media_url="http://www.example.com/cheeseburger.png")
    )

    print(message.sid)
