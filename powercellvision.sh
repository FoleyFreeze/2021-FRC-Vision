#!/bin/bash
#
cd /home/pi/2020-FRC-Vision
source /home/pi/.virtualenvs/cv/bin/activate
raspistill -o starting_picture.jpg
sleep 3
python3 vision.py
echo "ending vision"
sleep 2
exit

