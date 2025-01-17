#include "calib_stereo.h"
#include "calib_mono.h"

#include <random>

using namespace cv;
using namespace std;

void stereo_calibrate(std::string pathToImages, std::string filename, CalibParams& params){

    // replacing X in the camera name by 0 or 1.
    string left_images,right_images,left_fname,right_fname;
    unsigned found = params.cam_name.find_last_of("X");
    if(!found){
        cerr << "[Calibration stereo] could not identify variable X in camera name." << endl;
        exit(-1);
    }

    if(unsigned dot=filename.find_last_of(".")){
        left_fname = filename.substr(0,dot)+"_left.yml";
        right_fname = filename.substr(0,dot)+"_right.yml";
    }
    else{
        cerr << "[Calibration stereo] wrong yml file name." << endl;
        exit(-1);
    }

    /**** Mono calibration for each camera ****/
    if(!params.skip_mono_calib){
        params.cam_name = params.cam_name.substr(0,found) + "0" + params.cam_name.substr(found+1,params.cam_name.size());
        mono_calibrate(pathToImages,left_fname,params);
        params.cam_name = params.cam_name.substr(0,found) + "1" + params.cam_name.substr(found+1,params.cam_name.size());
        mono_calibrate(pathToImages,right_fname,params);
        params.cam_name = params.cam_name.substr(0,found) + "X" + params.cam_name.substr(found+1,params.cam_name.size());
    }

    /**** reading intrinsic parameters ****/

    Mat K0,K1,D0,D1,R,T,E,F;
    FileStorage leftFile(left_fname, 0);
    FileStorage rightFile(right_fname, 0);
    if(leftFile.isOpened() && rightFile.isOpened()){
        leftFile["K"] >> K0;leftFile["D"] >> D0;
        rightFile["K"] >> K1;rightFile["D"] >> D1;
        leftFile.release();rightFile.release();
    }else{
        cerr << "[error] couldn't open intrinsic params file! exiting..." << endl;
        exit(-1);
    }

    /**** find stereo corresponding features ****/
    vector<Point3f> structureBoard;
    vector<int> left_idx,right_idx;
    vector<Mat> l_rvecs,l_tvecs,r_rvecs,r_tvecs;
    vector<vector<Point2f>> left_stereoPts,right_stereoPts;
    //detecting pattern in all images
    findPatternStereo(pathToImages,left_stereoPts,right_stereoPts,params);
    //selecting the maximum nb of images
    std::mt19937 rng(0xFFFFFFFF);
    while(left_stereoPts.size() > params.MAX_IMAGES){
        std::uniform_int_distribution<uint32_t> uniformDistro(0,left_stereoPts.size());
        int idx = uniformDistro(rng);
        left_stereoPts.erase(left_stereoPts.begin()+idx);
        right_stereoPts.erase(right_stereoPts.begin()+idx);
    }
    cout << "[Calibration stereo] " << left_stereoPts.size() << " stereo points found!" << endl;
    calcPatternPosition(structureBoard,params);
    vector<vector<Point3f>> stereo_objectPoints(left_stereoPts.size(),structureBoard);

    /**** run stereo calibration *****/
    stereoCalibrate(stereo_objectPoints,left_stereoPts,right_stereoPts,K0,D0,K1,D1,params.image_size,R,T,E,F, CALIB_USE_INTRINSIC_GUESS | CALIB_FIX_INTRINSIC);

    std::cout << "[Calibration stereo] K0: " << endl << K0 << endl;
	std::cout << "[Calibration stereo] D0: " << endl << D0 << endl;
    std::cout << "[Calibration stereo] K1: " << endl << K1 << endl;
	std::cout << "[Calibration stereo] D1: " << endl << D1 << endl;
	std::cout << "[Calibration stereo] R: " << endl << R << endl;
	std::cout << "[Calibration stereo] T: " << endl << T << endl;

	/**** saving calibration params ****/
    FileStorage paramsFile(filename, 1);
	paramsFile << "K0" << K0 << "D0" << D0 << "K1" << K1 << "D1" << D1 << "R" << R << "T" << T;
    paramsFile.release();

    cout << "[Calibration stereo] calibration parameters written correctly." << endl;
}

void stereo_rectify(std::string pathToImages, std::string rectFolder, std::string filename, CalibParams& params){

    /**** read stereo params ****/
    Mat K0,K1,D0,D1,R,T;
    FileStorage intFile(filename, 0);
	unsigned dot = filename.find_last_of(".");
    FileStorage paramsFile(filename.substr(0,dot)+"_stereo.yml", 1);
    intFile["K0"] >> K0;
	intFile["D0"] >> D0;
	intFile["K1"] >> K1;
	intFile["D1"] >> D1;
	intFile["R"] >> R;
	intFile["T"] >> T;

    /**** compute and save stereo rectified parameters ****/
	Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];
    cv::stereoRectify(K0, D0, K1, D1, params.image_size, R, T, R1, R2, P1, P2, Q,CALIB_ZERO_DISPARITY,0, params.image_size, &validRoi[0], &validRoi[1]);
	cout << "[Rectification stereo] P1: " << endl << P1 << endl;
	cout << "[Rectification stereo] P2: " << endl << P2 << endl;

    if( paramsFile.isOpened() )
    {
        paramsFile << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        paramsFile.release();
    }
    else{
		std::cerr << "[error] can not save the calibration parameters\n";
		exit(-1);
    }

    /**** rectify and display the images ****/

    VideoCapture lcap,rcap;
    unsigned found = params.cam_name.find_last_of("X");
    unsigned percent =  params.cam_name.find_last_of("%");
    if(found){
        lcap.open(pathToImages + "/" + params.cam_name.substr(0,found) + "0" + params.cam_name.substr(found+1,params.cam_name.size()));
        rcap.open(pathToImages + "/" + params.cam_name.substr(0,found) + "1" + params.cam_name.substr(found+1,params.cam_name.size()));
        params.image_size.height = lcap.get(CAP_PROP_FRAME_HEIGHT);params.image_size.width = lcap.get(CAP_PROP_FRAME_WIDTH);
    }else{

        cerr << "[Rectification stereo] could not identify variable X in camera name." << endl;
        exit(-1);
    }

    Mat rmap[2][2];
    cv::initUndistortRectifyMap(K0, D0, R1, P1, params.image_size, CV_16SC2, rmap[0][0], rmap[0][1]);
    cv::initUndistortRectifyMap(K1, D1, R2, P2, params.image_size, CV_16SC2, rmap[1][0], rmap[1][1]);

	cv::Mat img[2],rimg;
    lcap >> img[0];
    rcap >> img[1];
    while(!img[0].empty() && !img[1].empty()){
        for(int k = 0; k < 2; k++ )
        {
            remap(img[k], rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
            Mat roi(rimg, validRoi[k]); //final rectified image
            stringstream ss;
            ss << rectFolder << params.cam_name.substr(0,found) << k << params.cam_name.substr(found+1,percent-(found+1)) << setw(5) << setfill('0') << lcap.get(CAP_PROP_POS_FRAMES) << "_rec.png";
            string nameimg = ss.str();
            std::cout << nameimg.c_str() << "\r";std::cout.flush();
            if(params.display){
                Mat test = rimg.clone();
				if(test.channels() == 1)
	                cvtColor(test,test,COLOR_GRAY2RGB);
                for(int i = 10;i <test.rows;i+=10)
                    line(test,cv::Point(0,i),cv::Point(test.cols-1,i),cv::Scalar(200,200,0));

                if(k){
                    line(img[k],cv::Point(K1.at<double>(0,2),0),cv::Point(K1.at<double>(0,2),params.image_size.height),cv::Scalar(0,255,0));
                    line(img[k],cv::Point(0,K1.at<double>(1,2)),cv::Point(params.image_size.width,K1.at<double>(1,2)),cv::Scalar(0,255,0));
                    imshow("right",test);
                    imshow("orignalR",img[k]);
                }
                else{
                    line(img[k],cv::Point(P1.at<double>(0,2),0),cv::Point(P1.at<double>(0,2),params.image_size.height),cv::Scalar(0,255,0));
                    line(img[k],cv::Point(0,P1.at<double>(1,2)),cv::Point(params.image_size.width,P1.at<double>(1,2)),cv::Scalar(0,255,0));
                    imshow("left",test);
                    imshow("originalL",img[k]);
                }
            }else
                imwrite(nameimg.c_str(),rimg); //create the image using Matrix roi

        }
        char k = waitKey(params.display?0:100);
        if(k == 'q')
            break;
        lcap >> img[0];
        rcap >> img[1];
    }
}


void stereo_calibrateAndRectify(string pathToImages, string rectFolder, string paramsFile, CalibParams& params){
    stereo_calibrate(pathToImages,paramsFile,params);
    stereo_rectify(pathToImages,rectFolder,paramsFile,params);
}
