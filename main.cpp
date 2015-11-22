#include <iostream>
#include <chrono>
#include <stdio.h>
#include <cstdlib>
#include <string.h>
#include <fstream>
#include <dirent.h>
#include <string.h>

#include <opencv2/opencv.hpp>

#include <../AAMlib/icaam.h>
#include <../AAMlib/sicaam.h>
#include <../AAMlib/wsicaam.h>
#include <../AAMlib/robustaam.h>
#include <../AAMlib/trainingdata.h>

#define WINDOW_NAME "AAM-Example"

using namespace std;
using namespace cv;

//Choose the fitting algorithm by using a different class
RobustAAM aam;

//Parameters for the fitting
int numParameters = 15;          //Number of used shape parameters
float fitThreshold = 0.001f;      //Termination condition

vector<string> descriptions;
Mat groups;

//Loads training data from a file and adds it the AAM
void loadTrainingData(string fileName) {
    cout<<"Load "<<fileName<<endl;
    TrainingData t;
    t.loadDataFromFile(fileName);

    Mat p = t.getPoints();
    Mat i = t.getImage();

    if(descriptions.empty()) {
        descriptions = t.getDescriptions();
        groups = t.getGroups();
    }

    aam.addTrainingData(p, i);
}

//Loads training data from a directory and adds it the AAM
void loadTrainingDataFromDir(string dirName){
    DIR *pDIR;
    struct dirent *entry;
    if((pDIR=opendir(dirName.c_str())) != NULL) {
        while((entry = readdir(pDIR)) != NULL) {
            //if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 )
            //string fileType = to_string(".xml");
            //string fileName = str(entry->d_name);
            if(strstr(entry->d_name, ".xml")) {
                loadTrainingData(dirName+entry->d_name);
            }
        }
        closedir(pDIR);
    }
}

//Draws the shape points on the image
Mat drawShape(Mat image, Mat points) {
    if(!aam.triangles.empty()) {
        for(int i=0; i<aam.triangles.rows; i++) {
            Point a,b,c;
            a = aam.getPointFromMat(points, aam.triangles.at<int>(i,0));
            b = aam.getPointFromMat(points, aam.triangles.at<int>(i,1));
            c = aam.getPointFromMat(points, aam.triangles.at<int>(i,2));

            line(image, a, b, Scalar(255,0,255),1);
            line(image, a, c, Scalar(255,0,255),1);
            line(image, b, c, Scalar(255,0,255),1);
        }
    }

    return image;
}

int main()
{
    string filePath_train = "/home/lucas/Radboud/090/annotated/";
    string filePath_test = "/home/lucas/Radboud/090/male/Test_Occlusion/";

    loadTrainingDataFromDir(filePath_train);

    //optional: Set the variance of the training data represented by the Shape/Appearance Parameters
    aam.setTargetShapeVariance(0.95);
    aam.setTargetAppVariance(0.95);

    //optional: Enable the preprocessing of images and set the used method to add robustness to the fitting
    aam.setPreprocessImages(true);
    aam.setProcessingMethod(AAM_PREPROC_DISTANCEMAPS);

    //optional: Set the used error function to add robustness in case of occluded faces, only works with classes RobustAAM and WSICAAM
    aam.setErrorFunction(AAM_ERR_TUKEY);

    //Train aam with Training Data
    aam.train();

    //Load the image the AAM should be fit to
    Mat fittingImage = imread(filePath_test+"Rafd090_10_Caucasian_male_neutral_frontal.jpg");
    if(!fittingImage.data) {
       cout<<"Could not load image"<<endl;
       return -1;
    }

    Mat image = fittingImage.clone();

    //Load image and initialize the fitting shape
    aam.setFittingImage(fittingImage);   //Converts image to right format
    aam.resetShape();    //Uses Viola-Jones Face Detection to initialize shape

    //Initialize with value > fitThreshold to enter the fitting loop
    float fittingChange = 20.0f;
    int steps = 0;

    //Terminate until fitting parameters change under predefined threshold
    // or 100 update steps have been executed
    while(fittingChange > fitThreshold && steps < 100) {
        fittingChange = aam.fit();   //Execute single update step
        steps++;
        cout<<"Step "<<steps<<" || Error per pixel: "<<aam.getErrorPerPixel()<<" || Parameter change: "<<fittingChange<<endl;

        Mat image = fittingImage.clone();
        Mat p = aam.getFittingShape();
        image = drawShape(image, p);
        imshow(WINDOW_NAME, image);

        waitKey(0); //Execute next step only when key pressed
    }

    //Draw the final triangulation and display the result
    //Mat image = fittingImage.clone();
    Mat p = aam.getFittingShape();

    image = drawShape(image, p);
    imshow(WINDOW_NAME, image);

    //Save the result
    TrainingData tr;
    tr.setImage(fittingImage);
    tr.setPoints(p);
    tr.setGroups(groups);
    tr.setDescriptions(descriptions);
    tr.saveDataToFile("out.xml");

    waitKey(0);

    return 0;
}

