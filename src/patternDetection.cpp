#include "patternDetection.h"

using namespace cv;

/**** Global variables for pattern detection ****/
static Point2f meanPT;
static Point2f p1,p2;
static std::vector<Point2f> ccenters,cornersP;
/************************************************/

//! Sorting function. Sorts keypoints by distance to the centre of the pattern.
bool sortKpt(KeyPoint kpt1, KeyPoint kpt2){return (norm(kpt1.pt-meanPT) < norm(kpt2.pt-meanPT));}
//!Sorting function. Sorts features by distance to the centre of the pattern.
bool sortPt(Point2f pt1, Point2f pt2){return (norm(pt1-meanPT) < norm(pt2-meanPT));}

//! Sorting function. Sorts corner features by their angle
bool sortAngle(Point2f kpt1, Point2f kpt2){
    Point2f pt1 = kpt1-meanPT, pt2 = kpt2-meanPT;
    float det1 = pt1.x*p1.y-(pt1.y*p1.x), det2 = pt2.x*p1.y-(pt2.y*p1.x);
    return ( (det1>0?1:-1) * acos((pt1.x*p1.x+pt1.y*p1.y)/(norm(pt1)*norm(p1))) < (det2>0?1:-1) * acos((pt2.x*p1.x+pt2.y*p1.y)/(norm(pt2)*norm(p1))));
}

//! Sorting function. Sorts the features belonging to the pyramid by their angle.
bool sortAnglePyr(Point2f kpt1, Point2f kpt2){
    Point2f pt1 = kpt1-p2, pt2 = kpt2-p2;
    float det1 = pt1.x*p1.y-(pt1.y*p1.x), det2 = pt2.x*p1.y-(pt2.y*p1.x);
    return ( (det1>0?1:-1) * acos((pt1.x*p1.x+pt1.y*p1.y)/(norm(pt1)*norm(p1))) < (det2>0?1:-1) * acos((pt2.x*p1.x+pt2.y*p1.y)/(norm(pt2)*norm(p1))));
}

//! Sorting function. Helps to find the top of the pyramid
bool findOrigin(const Point2f& pt1, const Point2f& pt2){
    float min = 1000;
    bool minIdx=true;
    for(unsigned int i=0;i<ccenters.size();i++){
        if(norm(pt1-ccenters[i]) < min){
         min = norm(pt1-ccenters[i]);
         minIdx =true;
        }
        if(norm(pt2-ccenters[i]) < min){
         min = norm(pt2-ccenters[i]);
         minIdx =false;
        }
    }
    return !minIdx;
}

//! refine blob centre detection. It is assumed that 2D feature follow a normal distribution. It does not take into account distortions due to projections.
void refineCentreDetection(const Mat& img, std::vector<KeyPoint>& keypoints){

    for(KeyPoint& kp : keypoints){
        // defining window around feature
        int win_size = 2*kp.size;
        Rect win_rect(kp.pt.x-win_size/2,kp.pt.y-win_size/2,win_size+1,win_size+1);
        if(!(win_rect.x >0) || !(win_rect.y > 0) || win_rect.x+win_size+1>img.cols-1 || win_rect.y+win_size+1>img.rows-1) //if window is out of boundaries skip refinement
            continue;
        Mat window = img(win_rect);
        // computing of pixels sample
        double x_mean=0,y_mean=0,weight_x=0,weight_y=0;
        for(int i=0;i<win_size;i++){
                x_mean += sum(window.col(i))[0] * (i+1);
                y_mean += sum(window.row(i))[0] * (i+1);
                weight_x += sum(window.col(i))[0];
                weight_y += sum(window.row(i))[0];
            }
            x_mean /= weight_x;
            y_mean /= weight_y;
        // -0.5 to centre pixels and because started at 1
        kp.pt.x = win_rect.x-0.5+x_mean;
        kp.pt.y = win_rect.y-0.5+y_mean;
    }
}

void calcPatternPosition(std::vector<Point3f>& corners,CalibParams& params)
{
    corners.clear();

    switch(params.calib_pattern)
    {
        case Pattern::CHESSBOARD:
          std::cout<<"Here"<<std::endl;
        case Pattern::CIRCLES_GRID:
          for( int i = 0; i < params.board_sz.height; ++i )
            for( int j = 0; j < params.board_sz.width; ++j )
                corners.push_back(Point3f(float( j*params.element_size ), float( i*params.element_size ), 0));
          break;

        case Pattern::ASYMMETRIC_CIRCLES_GRID:
          for( int i = 0; i < params.board_sz.height; i++ )
             for( int j = 0; j < params.board_sz.width; j++ )
                corners.push_back(Point3f(float((2*j + i % 2)*params.element_size), float(i*params.element_size), 0));
          break;

        case Pattern::PYRAMID:
            corners.push_back(Point3f(0.450,0.240,0));
            corners.push_back(Point3f(0.375,0.150,0));
            corners.push_back(Point3f(0.300,0.240,0));
            corners.push_back(Point3f(0.375,0.330,0));
            corners.push_back(Point3f(0.525,0.150,0));
            corners.push_back(Point3f(0.225,0.150,0));
            //bottom right
            corners.push_back(Point3f(0.600,0.000,0));
            corners.push_back(Point3f(0.750,0.000,0));
            corners.push_back(Point3f(0.750,0.150,0));
            //top right
            corners.push_back(Point3f(0.750,0.330,0));
            corners.push_back(Point3f(0.750,0.480,0));
            corners.push_back(Point3f(0.600,0.480,0));
            //top left
            corners.push_back(Point3f(0.150,0.480,0));
            corners.push_back(Point3f(0.000,0.480,0));
            corners.push_back(Point3f(0.000,0.330,0));
            //bottom left
            corners.push_back(Point3f(0.000,0.150,0));
            corners.push_back(Point3f(0.000,0.000,0));
            corners.push_back(Point3f(0.150,0.000,0));
            break;
    }
}

void drawPyramidPattern(Mat& img, std::vector<Point2f>& centers, bool found){

    line(img, meanPT, meanPT+Point2f(0,10), Scalar(0,255,0), 1, LINE_AA);
    line(img, meanPT, meanPT+Point2f(10,0), Scalar(0,255,0), 1, LINE_AA);
    line(img, meanPT, meanPT+p1, Scalar(255,255,255), 1, LINE_AA);
    if(ccenters.size() > 0){
        circle(img,ccenters[0],3,Scalar(255,255,255));
        circle(img,ccenters[1],3,Scalar(255,255,255));
        circle(img,ccenters[2],3,Scalar(255,255,255));
        circle(img,ccenters[3],3,Scalar(255,255,255));
    }
    else
        std::cerr << "not enough clusters!" << std::endl;
    if(found){
        line(img, centers[0], centers[3], Scalar(0,0,255),2);
        line(img, centers[0], centers[2], Scalar(0,0,255),2);
        line(img, centers[3], centers[2], Scalar(0,0,255),2);
        line(img, centers[1], centers[0], Scalar(0,0,255),2);
        line(img, centers[4], centers[1], Scalar(0,0,255),2);
        line(img, centers[4], centers[0], Scalar(0,0,255),2);
        line(img, centers[1], centers[2], Scalar(0,0,255),2);
        line(img, centers[5], centers[1], Scalar(0,0,255),2);
        line(img, centers[5], centers[2], Scalar(0,0,255),2);

        line(img, centers[6], centers[7], Scalar(255,0,0),2);
        line(img, centers[7], centers[8], Scalar(255,0,0),2);
        line(img, centers[6], centers[8], Scalar(255,0,0),2);

        line(img, centers[9], centers[10], Scalar(0,255,0),2);
        line(img, centers[10], centers[11], Scalar(0,255,0),2);
        line(img, centers[9], centers[11], Scalar(0,255,0),2);

        line(img, centers[12], centers[13], Scalar(255,0,0),2);
        line(img, centers[13], centers[14], Scalar(255,0,0),2);
        line(img, centers[12], centers[14], Scalar(255,0,0),2);

        line(img, centers[15], centers[16], Scalar(0,255,0),2);
        line(img, centers[16], centers[17], Scalar(0,255,0),2);
        line(img, centers[15], centers[17], Scalar(0,255,0),2);

    }
        for( size_t i = 0; i < centers.size(); i++ ){
        Scalar colour(i*10,0,255-(i*10));
        if(i<6)
            circle(img, centers[i], 1, colour, -1, 8, 0 );
        else
            circle(img, centers[i], 1, colour, -1, 8, 0 );

        std::stringstream ss; ss << i;
        putText(img,ss.str(),centers[i],0,0.5,Scalar(0,255,0));
    }
}

bool findPyramid(const Mat& img, std::vector<Point2f>& centers){

    centers.clear();
    /**** Blob detection ****/
    SimpleBlobDetector::Params params;
    params.minThreshold = 0;
    params.maxThreshold = 255;
    params.filterByColor = true;
    params.blobColor = 255;

    params.filterByArea = true;
    params.minArea = 0;
    params.maxArea = 50;
    params.filterByCircularity = true;
    params.minCircularity = 0.75;
    params.filterByConvexity = true;
    params.minConvexity = 0.87;
    // Filter by Inertia
    params.filterByInertia = true;
    params.minInertiaRatio = 0.2;
    Ptr<SimpleBlobDetector> blobdet = SimpleBlobDetector::create(params);

    std::vector<KeyPoint> keypoints;
    blobdet->detect( img, keypoints);

    std::cout << keypoints.size() << " keypoints \r ";std::cout.flush();
    if(keypoints.size() != 18)
        return false;

    refineCentreDetection(img,keypoints);

    /**** finding center of the pattern ****/
     meanPT = Point2f(0,0);

     for( unsigned int i = 0; i < keypoints.size(); i++ )
    {
         Point2f center(keypoints[i].pt.x, keypoints[i].pt.y);
         centers.push_back(keypoints[i].pt);
         meanPT += center;
    }
    meanPT/=(int)(keypoints.size());

    /**** feature identification ****/

    // sorting feature by their distance to the center
    std::sort(centers.begin(),centers.end(),sortPt);

    // center position refinement with only features from the center of the pyramid
    meanPT = Point2f(0,0);
    for(uint i=0;i<3;i++)
        meanPT += centers[i];
    meanPT/=3;
    // selecting corner features
    cornersP.clear();
    cornersP.insert(cornersP.end(),centers.begin()+6,centers.end());
    Mat labels,ccenters_;
    //kmeans to cluster the 4 corners
    kmeans(cornersP,4,labels,TermCriteria(TermCriteria::EPS,50,0.1),20,KMEANS_RANDOM_CENTERS,ccenters_);
    ccenters.clear();
    //computing the center of each corner
    for(unsigned int k=0;k<4;k++)
        ccenters.push_back(Point2f(ccenters_.at<float>(k,0),ccenters_.at<float>(k,1)));

    //finding the origin of the pyramid (top summit)
    std::sort(centers.begin()+3,centers.begin()+6,findOrigin);
    p1 = centers[3] - (centers[4]+centers[5])/2;
    p2 = (centers[4]+centers[5])/2 - p1;
    //sorting features by their angle
    sort(centers.begin(),centers.begin()+3,sortAnglePyr);
    sort(centers.begin()+4,centers.begin()+6,sortAnglePyr);
    sort(centers.begin()+6,centers.end(),sortAngle);

    /**** consistency check ****/

    bool corner1 = (norm(centers[6]-centers[7])+norm(centers[7]-centers[8])+norm(centers[8]-centers[6]))/3 < 130;
    bool corner2 = (norm(centers[9]-centers[10])+norm(centers[10]-centers[11])+norm(centers[11]-centers[9]))/3 < 130;
    bool corner3 = (norm(centers[12]-centers[13])+norm(centers[13]-centers[14])+norm(centers[14]-centers[12]))/3 < 130;
    bool corner4 = (norm(centers[15]-centers[16])+norm(centers[16]-centers[17])+norm(centers[17]-centers[15]))/3 < 130;
    bool ratio1 = norm(centers[3]-centers[0])/norm(centers[3]-centers[4]) > 0.45 && norm(centers[3]-centers[0])/norm(centers[3]-centers[4]) < 0.55;
    bool ratio2 = norm(centers[3]-centers[2])/norm(centers[3]-centers[5]) > 0.45 && norm(centers[3]-centers[2])/norm(centers[3]-centers[5]) < 0.55;
    bool ratio3 = norm(centers[2]-centers[0])/norm(centers[4]-centers[5]) > 0.45 && norm(centers[2]-centers[0])/norm(centers[4]-centers[5]) < 0.55;

    return ratio1 && ratio2 && ratio3 && corner1 && corner2 && corner3 && corner4;
}

std::vector<std::vector<Point2f>> findPattern(const std::string& pathToImages, std::vector<int>& image_idx, CalibParams& params){

    VideoCapture cap(pathToImages+"/"+params.cam_name);
	std::cout << "[Calibration] opening " << pathToImages+"/"+params.cam_name << std::endl;
    std::vector<std::vector<Point2f>> imagePoints;
    image_idx.clear();
    namedWindow("calibration",WINDOW_NORMAL);
    if(!cap.isOpened()){
        std::cerr << "could not open video or find images. exiting..." << std::endl;
        exit(-1);
    }else{
        params.image_size.height = cap.get(CAP_PROP_FRAME_HEIGHT);params.image_size.width = cap.get(CAP_PROP_FRAME_WIDTH);}

    Mat img, rimg;
    cap >> img;
    while(!img.empty()){

        /**** reading or acquiring the current frame ***/
        if(img.channels()>1){
            rimg = img.clone();
            cvtColor(img, img, COLOR_BGR2GRAY);
        }else
            cvtColor(img, rimg, COLOR_GRAY2BGR);

        /**** detecting the calibration pattern ****/
        std::vector<Point2f> corners;
        bool found=false;
        switch(params.calib_pattern){
            case Pattern::CHESSBOARD:
                found = findChessboardCorners(rimg, params.board_sz, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);break;
            case Pattern::ASYMMETRIC_CIRCLES_GRID:
                found = findCirclesGrid(rimg, params.board_sz, corners,CALIB_CB_ASYMMETRIC_GRID);break;
            case Pattern::CIRCLES_GRID:
                found = findCirclesGrid(rimg, params.board_sz, corners,CALIB_CB_SYMMETRIC_GRID);break;
            case Pattern::PYRAMID:
                found = findPyramid(img, corners);break;
        }

        if(found){ // if the pattern could not been found, display the image and continue to the next frame
            /**** displaying the pattern ****/
            switch(params.calib_pattern){
                case Pattern::CHESSBOARD:
                    drawChessboardCorners(rimg, params.board_sz,corners, found);break;
                case Pattern::ASYMMETRIC_CIRCLES_GRID:
                    drawChessboardCorners(rimg, params.board_sz,corners, found);break;
                case Pattern::CIRCLES_GRID:
                    drawChessboardCorners(rimg, params.board_sz,corners, found);break;
                case Pattern::PYRAMID:
                    drawPyramidPattern(rimg,corners, found);break;
            }

            imshow("calibration", rimg);
            char k;
            k = waitKey(100);
            imagePoints.push_back(corners);
            image_idx.push_back(cap.get(CAP_PROP_POS_FRAMES));
            if(k == 'c') // if c is pressed, stop the acquisition
                break;
        }else{
        //                cerr << "chessboard not found" << endl;
            imshow("calibration", rimg);
            waitKey(100);
        }


        for(int i=0;i<params.interval;i++)
            cap >> img;
    }
	cap.release();
	destroyAllWindows();
    return imagePoints;
}

void findPatternStereo(const std::string& pathToImages, std::vector<std::vector<Point2f>>& leftPoints, std::vector<std::vector<Point2f>>& rightPoints, CalibParams& params){

    leftPoints.clear();rightPoints.clear();
    namedWindow("cleft",WINDOW_NORMAL);namedWindow("cright",WINDOW_NORMAL);
    VideoCapture cap_left,cap_right;

    if(unsigned found = params.cam_name.find_last_of("X")){
        cap_left.open(pathToImages + "/" + params.cam_name.substr(0,found) + "0" + params.cam_name.substr(found+1,params.cam_name.size()));
        cap_right.open(pathToImages + "/" + params.cam_name.substr(0,found) + "1" + params.cam_name.substr(found+1,params.cam_name.size()));
        params.image_size.height = cap_left.get(CAP_PROP_FRAME_HEIGHT);params.image_size.width = cap_left.get(CAP_PROP_FRAME_WIDTH);
    }else{
        std::cerr << "[Calibration] could not identify variable X in camera name." << std::endl;
        exit(-1);
    }

    Mat limg, rimg,climg,crimg;
    cap_left >> limg;
    cap_right >> rimg;
    while(!limg.empty() && !rimg.empty()){
        /**** reading images ****/
        if(limg.channels()>1){
            climg = limg.clone();
            cvtColor(limg, limg, COLOR_BGR2GRAY);
        }else
            cvtColor(limg, climg, COLOR_GRAY2BGR);
        if(rimg.channels()>1){
            crimg = rimg.clone();
            cvtColor(rimg, rimg, COLOR_BGR2GRAY);
        }else
            cvtColor(rimg, crimg, COLOR_GRAY2BGR);

        std::vector<Point2f> lcorners,rcorners;
        bool lfound=false,rfound=false;
        /**** detecting the calibration pattern ****/
        switch(params.calib_pattern){
            case Pattern::CHESSBOARD:
                lfound = findChessboardCorners(limg, params.board_sz, lcorners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
                rfound = findChessboardCorners(rimg, params.board_sz, rcorners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);break;
            case Pattern::ASYMMETRIC_CIRCLES_GRID:
                lfound = findCirclesGrid(limg, params.board_sz, lcorners,CALIB_CB_ASYMMETRIC_GRID);
                rfound = findCirclesGrid(rimg, params.board_sz, rcorners,CALIB_CB_ASYMMETRIC_GRID);break;
            case Pattern::CIRCLES_GRID:
                lfound = findCirclesGrid(limg, params.board_sz, lcorners,CALIB_CB_SYMMETRIC_GRID);
                rfound = findCirclesGrid(rimg, params.board_sz, rcorners,CALIB_CB_SYMMETRIC_GRID);break;
            case Pattern::PYRAMID:
                lfound = findPyramid(limg, lcorners);
                rfound = findPyramid(rimg, rcorners);break;
        }

        if(lfound && rfound){ // if a calibration pattern is not detected, continue
            /**** displaying the calibration pattern ****/
            switch(params.calib_pattern){
                case Pattern::CHESSBOARD:
                    drawChessboardCorners(climg, params.board_sz,lcorners, lfound);
                    drawChessboardCorners(crimg, params.board_sz,rcorners, rfound);break;
                case Pattern::ASYMMETRIC_CIRCLES_GRID:
                    drawChessboardCorners(climg, params.board_sz,lcorners, lfound);
                    drawChessboardCorners(crimg, params.board_sz,rcorners, rfound);break;
                case Pattern::CIRCLES_GRID:
                    drawChessboardCorners(climg, params.board_sz,lcorners, lfound);
                    drawChessboardCorners(crimg, params.board_sz,rcorners, rfound);break;
                case Pattern::PYRAMID:
                    drawPyramidPattern(climg,lcorners, lfound);
                    drawPyramidPattern(crimg,rcorners, rfound);break;
            }

            imshow("cleft", climg);
            imshow("cright", crimg);
            char k;
            k = waitKey(100);
            leftPoints.push_back(lcorners);
            rightPoints.push_back(rcorners);
            if(k == 'c') // if c is pressed, stop the detection
                break;
        }else{
            imshow("cleft", climg);
            imshow("cright", crimg);
            waitKey(100);
        }

        for(int i=0;i<params.interval;i++){
            cap_left >> limg;
            cap_right >> rimg;
        }
    }
    destroyAllWindows();
}

double computeMRE(const std::vector<std::vector<Point3f>>& objectPoints, const std::vector<std::vector<Point2f>>& imagePts, const std::vector<Mat>& rvecs, const std::vector<Mat>& tvecs, const Mat& K, const Mat& dist, std::vector<double>& repValues){

    std::vector<Point2f> reprojPts;
    repValues.clear();
    double mre=0;
    for(unsigned int p=0;p<objectPoints.size();p++){
        double mre_=0;
        projectPoints(objectPoints[p],rvecs[p],tvecs[p],K,dist,reprojPts);
        for(unsigned int q=0;q<reprojPts.size();q++)
            mre_ += sqrt(pow(imagePts[p][q].x-reprojPts[q].x,2)+pow(imagePts[p][q].y-reprojPts[q].y,2));
        mre_ /= reprojPts.size();
        repValues.push_back(mre_);
        mre += mre_;
    }
    mre /= objectPoints.size();
    return mre;
}
