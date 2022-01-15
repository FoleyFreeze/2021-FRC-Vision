#!/bin/bash
#
cd /home/pi/2020-FRC-Vision
echo "starting vision"
raspistill -o cam.jpg
source /home/pi/.virtualenvs/cv/bin/activate
python3 vision.py
echo "ending vision"
sleep 3
exit

