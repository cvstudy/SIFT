#if 0

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    const Mat inputImg = imread("/Users/jisoo/Code/Opencv/sift/IMG_3363.jpg");
    Mat grayImg;
    Mat resizeImg;
    Mat H;
    vector<Point2f> obj,scene;
    vector<Point2f> refImgCorners(4);

    cvtColor(inputImg,grayImg, CV_BGR2GRAY);
    resize(grayImg, resizeImg, Size(inputImg.cols/6,inputImg.rows/6), 0, 0, CV_INTER_LINEAR);

    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
    vector<KeyPoint> keypoints;
    sift->detect(resizeImg,keypoints);
    //Mat dstImg;
    //drawKeypoints(resizeImg,keypoints,dstImg);
    //namedWindow("SIFT",CV_WINDOW_AUTOSIZE);

    //SIFT íŠ¹ì§•ì  ì„œìˆ ?
    Mat descriptions;
    sift->compute(resizeImg,keypoints,descriptions);

    VideoCapture stream1(0);
    refImgCorners[0] =Point2f(0,0);
    refImgCorners[1] =Point2f(resizeImg.cols,0);
    refImgCorners[2] =Point2f(resizeImg.cols,resizeImg.rows);
    refImgCorners[3] =Point2f(0,resizeImg.rows);

    //vector<DMatch> matcher = BFMatcher::BFMatcher.create();

    if(!stream1.isOpened()){
        cout << "cannot open camera";
    }


    while (true) {
        Mat cameraFrameOrg;
        //Mat cameraFrameresize;
        Mat cameraFrameGray;
        Mat trgdescriptions;
        vector<vector<DMatch>> matches;
        vector<KeyPoint> trgpoints;
        vector<Point2f> sceneCorners;
        vector<DMatch> good_matches;
        Mat dstImg;

        stream1.read(cameraFrameOrg);

        cvtColor(cameraFrameOrg, cameraFrameGray, CV_BGR2GRAY);
        sift->detect(cameraFrameGray,trgpoints);
        sift->compute(cameraFrameGray,trgpoints,trgdescriptions);

        //FLANN
        Ptr<cv::FlannBasedMatcher> matcher = FlannBasedMatcher::create();
       // FlannBasedMatcher matcher;
        if(descriptions.type()!=CV_32F) descriptions.convertTo(descriptions,CV_32F);
        if(trgdescriptions.type()!=CV_32F) trgdescriptions.convertTo(trgdescriptions,CV_32F);
        matcher->knnMatch(descriptions,trgdescriptions,matches,2);

        //DescriptorMatcher::knnMatch(descriptions,trgdescriptions,matches,k=2);
        for(int i = 0;i < min(trgdescriptions.rows-1,(int)matches.size());i++){
            if((matches[i][0].distance<0.6*(matches[i][1].distance))&&((int)matches[i].size() <= 2 && (int)matches[i].size() > 0)){
                good_matches.push_back(matches[i][0]);
            }
        }
        //good matches
        drawMatches(resizeImg,keypoints,cameraFrameGray,trgpoints,good_matches,dstImg);

        // ë°•ìŠ¤ì¹˜ê¸°
        if(good_matches.size()>10){
            for(unsigned long i = 0; i <good_matches.size(); i++){
                obj.push_back(keypoints[good_matches[i].queryIdx].pt);
                scene.push_back(trgpoints[good_matches[i].trainIdx].pt);
            }
            try
            {
                H = findHomography(obj,scene,CV_RANSAC);
            }
            catch (Exception e) {}

            perspectiveTransform(refImgCorners,sceneCorners, H);
            line(dstImg, sceneCorners[0] + Point2f(resizeImg.cols, 0), sceneCorners[1] + Point2f(resizeImg.cols, 0), Scalar(0, 255, 0), 4);
            line(dstImg, sceneCorners[1] + Point2f(resizeImg.cols, 0), sceneCorners[2] + Point2f(resizeImg.cols, 0), Scalar(0, 255, 0), 4);
            line(dstImg, sceneCorners[2] + Point2f(resizeImg.cols, 0), sceneCorners[3] + Point2f(resizeImg.cols, 0), Scalar(0, 255, 0), 4);
            line(dstImg, sceneCorners[3] + Point2f(resizeImg.cols, 0), sceneCorners[0] + Point2f(resizeImg.cols, 0), Scalar(0, 255, 0), 4);
        }

        imshow("SIFT",dstImg);
        good_matches.clear();

        if (waitKey(20)=='q')
            break;

    }



    return 0;
}

#endif

#if 1
// CV_example_180218.cpp: 콘솔 응용 프로그램의 진입점을 정의합니다.
//

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;

using namespace cv::xfeatures2d;

int main()
{
    //load training image
    Mat object = imread("IMG_3363.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    if (!object.data)
    {
        cout << "can't open image";
        return -1;
    }
    resize(object, object, Size(object.cols/6,object.rows/6), 0, 0, CV_INTER_LINEAR);

    namedWindow("Good Matches", CV_WINDOW_AUTOSIZE);

    //SURF detector, descriptor parameters
    int minHess;
    vector<KeyPoint> kpObject, kpImage;
    Mat desObject, desImage;

    //SURF detector, descriptor parameters and match object initialization
    minHess = 1000;
    Ptr<SurfFeatureDetector> detector = SurfFeatureDetector::create(minHess);
    detector->detect(object, kpObject);
    Ptr<SurfDescriptorExtractor> extractor = SurfDescriptorExtractor::create();
    extractor->compute(object, kpObject, desObject);

    //Initialize match object
    FlannBasedMatcher matcher;

    //Initialize video and display window
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        return -1;
    }

    //Object corner points for plotting box
    vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0, 0);
    obj_corners[1] = cvPoint(object.cols, 0);
    obj_corners[2] = cvPoint(object.cols, object.rows);
    obj_corners[3] = cvPoint(0, object.rows);

    //Video loop
    char escapeKey = 'k';
    double frameCount = 0;
    float thresholdMatchingNN = 0.6;
    unsigned int thresholdGoodMatches = 8;
    //unsigned int thresholdGoodMatchesV[] = { 4, 5,6, 7, 8, 9, 10 };

    cout << thresholdGoodMatches << endl;

    while (escapeKey != 'q')
    {
        Mat frame;
        Mat image(object.size(), object.type());
        //gram video fram
        cap >> frame;
        cvtColor(frame, image, CV_RGB2GRAY);

        Mat des_image, img_matches, H;
        vector<KeyPoint> kp_image;
        vector<vector<DMatch>> matches;
        vector<DMatch> good_matches;
        vector<Point2f> obj;
        vector<Point2f> scene;
        vector<Point2f> scene_corners(4);

        //detect interest points of query image
        detector->detect(image, kp_image);
        //extract interest point descriptors of query image
        extractor->compute(image, kp_image, des_image);

        //matching descriptors vectors using FLANN matcher

        if(desObject.type()!=CV_32F) desObject.convertTo(desObject,CV_32F);
        if(des_image.type()!=CV_32F) des_image.convertTo(des_image,CV_32F);
        matcher.knnMatch(desObject, des_image, matches, 2); //find the best 2 matches of each descriptor

        //select only good matches
        for (int i = 0; i < std::min(des_image.rows - 1, (int)matches.size()); i++) //THIS LOOP IS SENSITIVE TO SEGFAULTS
        {
            if ((matches[i][0].distance < thresholdMatchingNN * (matches[i][1].distance)) && ((int)matches[i].size() <= 2 && (int)matches[i].size() > 0))
            {
                good_matches.push_back(matches[i][0]);
            }
        }

        //Draw only "good" matches
        drawMatches(object, kpObject, image, kp_image, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        //Localize the object into scene image
        if (good_matches.size() >= thresholdGoodMatches)
        {

            //Display that the object is found
            putText(img_matches, "Object Found", cvPoint(10, 50), FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(0, 0, 250), 1, CV_AA);
            for (int i = 0; i < good_matches.size(); i++)
            {
                //Get the keypoints from the good matches
                obj.push_back(kpObject[good_matches[i].queryIdx].pt);
                scene.push_back(kp_image[good_matches[i].trainIdx].pt);
            }

            H = findHomography(obj, scene, CV_RANSAC);
            //cout << "H = " << H << endl;
            perspectiveTransform(obj_corners, scene_corners, H);

            //Draw lines between the corners (the mapped object in the scene image )
            line(img_matches, scene_corners[0] + Point2f(object.cols, 0), scene_corners[1] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
            line(img_matches, scene_corners[1] + Point2f(object.cols, 0), scene_corners[2] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
            line(img_matches, scene_corners[2] + Point2f(object.cols, 0), scene_corners[3] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
            line(img_matches, scene_corners[3] + Point2f(object.cols, 0), scene_corners[0] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
        }
        else
        {
            putText(img_matches, "", cvPoint(10, 50), FONT_HERSHEY_COMPLEX_SMALL, 3, cvScalar(0, 0, 250), 1, CV_AA);
        }

        //Show detected matches
        imshow("Good Matches", img_matches);
        good_matches.clear();

        escapeKey = cvWaitKey(1);

    }

    cap.release();
    return 0;
}

#endif
