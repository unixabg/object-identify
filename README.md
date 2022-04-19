# object-identify
Tool for object identification with opencv and python.

# Derived work
This code modified examples provided at https://core-electronics.com.au/guides/object-identify-raspberry-pi/

# License
This derived code is licensed as the original examples under https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

# Install (Debian 11)
1. Clone the repository
2. cd object-identify
3. sudo apt install python3-pip
4. pip install opencv-python
5. wget https://core-electronics.com.au/media/kbase/491/Object_Detection_Files.zip
6. unzip Object_Detection_Files.zip

# Run the code
python3 object-identify.py --video (/dev/video0 | test.mp4 | test.jpg)

# Platforms
Original example targeted Raspberry Pi. Targets for object-identify Debian based Linux.
Currently testing primarily on Debian 11


