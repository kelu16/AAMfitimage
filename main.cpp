#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <../AAMlib/icaam.h>
#include <../AAMlib/trainingdata.h>

#define WINDOW_NAME "AAM-Example"

using namespace std;
using namespace cv;

ICAAM aam;

//Parameters for the fitting
int numParameters = 15;          //Number of used shape parameters
float fitThreshold = 0.05f;      //Termination condition

vector<string> descriptions;
Mat groups;

void loadTrainingData(string fileName) {
    TrainingData t;
    t.loadDataFromFile(fileName);

    Mat p = t.getPoints();
    Mat i = t.getImage();

    if(descriptions.empty()) {
        descriptions = t.getDescriptions();
        groups = t.getGroups();
    }

    i.convertTo(i, CV_32FC3);
    cvtColor(i,i,CV_BGR2GRAY);

    aam.addTrainingData(p, i);
}

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
    string filePath_train = "/home/lucas/Dropbox/Diplomarbeit/Code/trainingdata/";
    string filePath_test = "/home/lucas/Dropbox/Diplomarbeit/Code/testimages/";

    loadTrainingData(filePath_train+"data1.xml");
    loadTrainingData(filePath_train+"data2.xml");
    loadTrainingData(filePath_train+"data3.xml");
    loadTrainingData(filePath_train+"data4.xml");
    loadTrainingData(filePath_train+"data5.xml");
    loadTrainingData(filePath_train+"data6.xml");
    loadTrainingData(filePath_train+"data7.xml");
    loadTrainingData(filePath_train+"data8.xml");
    loadTrainingData(filePath_train+"data9.xml");
    loadTrainingData(filePath_train+"data10.xml");
    loadTrainingData(filePath_train+"data11.xml");
    loadTrainingData(filePath_train+"data12.xml");
    loadTrainingData(filePath_train+"data13.xml");
    loadTrainingData(filePath_train+"data14.xml");
    loadTrainingData(filePath_train+"data15.xml");
    loadTrainingData(filePath_train+"data16.xml");
    loadTrainingData(filePath_train+"data17.xml");

    loadTrainingData(filePath_train+"Rafd090_03_Caucasian_male_angry_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_03_Caucasian_male_contemptuous_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_03_Caucasian_male_disgusted_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_03_Caucasian_male_fearful_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_03_Caucasian_male_happy_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_03_Caucasian_male_neutral_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_03_Caucasian_male_sad_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_03_Caucasian_male_surprised_frontal.xml");

    loadTrainingData(filePath_train+"Rafd090_07_Caucasian_male_angry_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_07_Caucasian_male_contemptuous_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_07_Caucasian_male_disgusted_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_07_Caucasian_male_fearful_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_07_Caucasian_male_happy_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_07_Caucasian_male_neutral_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_07_Caucasian_male_sad_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_07_Caucasian_male_surprised_frontal.xml");

    //Train aam with Training Data
    //optional: Set number of used Shape/Appearance Parameters
    aam.setNumShapeParameters(numParameters);
    aam.setNumAppParameters(numParameters);
    aam.train();

    namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);

    Mat fittingImage = imread(filePath_test+"8.jpg");

    //Load image and initialize the fitting shape
    aam.setFittingImage(fittingImage);   //Converts image to right format
    aam.resetShape();    //Uses Viola-Jones Face Detection to initialize shape


    //Initialize with value > fitThreshold to enter the fitting loop
    float fittingChange = 20.0f;
    int steps = 0;

    auto start = std::chrono::system_clock::now();

    //Terminate until fitting parameters change under predefined threshold
    // or 100 update steps have been executed
    while(fittingChange > fitThreshold && steps<100) {
        fittingChange = aam.fit();   //Execute single update step
        steps++;
        cout<<"Step "<<steps<<" || Error per pixel: "<<aam.getErrorPerPixel()<<" || Parameter change: "<<fittingChange<<endl;

        Mat image = fittingImage.clone();
        Mat p = aam.getFittingShape();
        image = drawShape(image, p);
        imshow(WINDOW_NAME, image);

        waitKey(0); //Execute next step only when key pressed
    }

    auto end = std::chrono::system_clock::now();
    auto duration = end-start;
    cout<<duration.count()/steps/(double) CLOCKS_PER_SEC<<endl;

    //Draw the final triangulation and display the result
    Mat image = fittingImage.clone();
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

