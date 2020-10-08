# SocialDistanceYOLORaspberry
Social Distance Monitoring using Raspberry Pi based on YOLOv3<br />
Tips:<br />
1- Download weights file from : https://pjreddie.com/media/files/yolov3-tiny.weights
<br />2- Add the downloaded file to src folder<br />
<br />
On a fresh installation of the Raspberry Pi 3 or 4 install:<br />

        sudo pip3 install imutils
        sudo pip3 install opencv-python opencv-contrib-python
        sudo apt-get install libhdf5-103
        sudo apt-get install libatlas-base-dev
        sudo apt-get install libjasper-dev
        sudo apt-get update
        sudo apt-get install libqtgui4
        sudo apt-get install libqt4-test
        sudo pip3 install scipy
<br />
<br />
Start social distance detection : <br />

        # LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1.2.0 python3 realtime_yolo_cpu.py
