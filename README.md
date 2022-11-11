This is an updated version of the calibration software for openCV4, the only changes have been to update with the new variable names, no other fixes or changes have been applied.

the -h results in a segmentation fault, otherise the software is working and the monocalibration was tested on the Imaging Source Camera.

useful resources from opencv on camera calibration:

    https://docs.opencv.org/4.5.5/d4/d94/tutorial_camera_calibration.html
    https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

################
Version Info

################

    version 1.0
    forked from: https://github.com/abeauvisage/uasl-calibration

################
DEPENDENCIES

################

    Opencv 4 minimum

################
INSTALLATION

################

    extract library folder
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    $ make doc (optional for generating documentation)

#########
USAGE

#########

lab_mono_calibration [IMAGEPATH] [IMAGEEXP] [YMLPATH] -[OPTIONS]

	./lab_mono_calibration ~/uasl-calibration_opencv4/raw frame_%08d.bmp ~/uasl-calibration/kmat -c -p chessboard -e 0.03

    ./lab_mono_calibration pathToImageFolder cam1_image%05d.png pathToYMLFile.yml | using a sequence of images
    ./lab_mono_calibration pathToImageFolder calibration_vid.mp4 pathToYMLFile.yml | using a video
    ./lab_mono_calibration /dev video0 pathToYMLFile.yml | using the camera

#####################
Supported options

#####################

    -H,-h display this help.
    -n specify the minimum number of images to stop the calibration.
    -m specify the MRE threshold to stop the calibration.
    -e set the size of the elements of the chessboard.
    -d display rectified images, otherwise save them.
    -c calibrate and rectify with the provided images.
    -i interval rate at which images are processed.
