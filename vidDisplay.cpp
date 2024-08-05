/*
    Sachin Palahalli Chandrakumar
    Spring 2024
    Code to capture video and pass the image frame required for training and classifying.
*/

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Defining the prototypes of methods used in this file
int applyThreshold(cv::Mat &src, cv::Mat &dst );
int cleanImage(cv::Mat &src, cv::Mat &dst);
int getRegionMap(cv::Mat &src, cv::Mat &regionMap, cv::Mat &dst);
int createFeatures(cv::Mat &regionMap, cv::Mat &allRegionImage, int totalRegions);
int getMomentsLibrary(cv::Mat &regionMap);
int getRegionMoments(cv::Mat &regionImage, cv::Mat regionMoments);
int getSingleRegionImage(cv::Mat &regionMap, cv::Mat &regionImage, int regionId);
int getRotatedImage(cv::Mat &regionImage, cv::Mat &rotatedRegion);
int processImageforRecognition(cv::Mat &src);
int processImageforRecognitionUsingDNNEmbeddings(cv::Mat &src);
int get_image_paths( std::vector<string> &image_paths);
int read_image_data_csv( char *filename, std::vector<char *> &filenames, std::vector<std::vector<float>> &data, int echo_file );
int classifyImage(cv::Mat &inputImage, std::vector<char*> objectNames, std::vector<std::vector<float>> objectFeatures);
int classifyImageDNNEmbeddings(cv::Mat &inputImage, std::vector<char*> objectNames, std::vector<std::vector<float>> trainedObjectFeatures);
int classifyImageKNN(cv::Mat &inputImage, std::vector<char*> objectNames, std::vector<std::vector<float>> objectFeatures, int k_neighbours);
int segmentImage(cv::Mat &src, cv::Mat &regionMap, cv::Mat &dst, std::vector<int> &regionIds, std::unordered_map<int, std::vector<int>> &regionsStatsMap);

/**
 * @brief reads all images csv files and gets the feature vector data.
 * This function take the path where the image feature data is present and then 
 * loads all the data present in the file to memory
 * @param[in] featuresFileName The csv file name.
 * @param[in] fileNames The list of all image names.
 * @param[in] data The 2D list of all images feature vector data.
 * @return 0.
*/
int getImagesFeaturesData(string featuresFileName, std::vector<char*> &fileNames, std::vector<std::vector<float>> &data){
    
    int echo_file = 0;
    read_image_data_csv( &featuresFileName[0], fileNames, data, echo_file );
    int totalFiles = fileNames.size();
    int totalData = data.size();
    if(totalFiles != totalData){
        cout<<" Number of Image names and the filter vectores present sould be same!";
        exit(-1);
    }
    return 0;
}


int main(int argc, char *argv[]){

    if(argc != 2){
        printf("Incorrect Command Line input. Usage: ");
        exit(-1);
    }

    // command line comments map
    std::unordered_map<string,int> command_map;
    command_map["train_knn"] = 1;
    command_map["train_dnn"] = 2;
    command_map["classify"] = 3;
    command_map["classify_dnn"] = 4;
    command_map["classify_knn"] = 5;
    

    int command = command_map[argv[1]];

    cv::Mat src;
    if(command == 1 || command == 2){
        std::vector<string> image_paths;
        get_image_paths(image_paths);
        for(int i=0;i<image_paths.size();i++){
            
            cout<<"Processing image :"<<image_paths[i]<<endl;
            //read the image
            src = imread(image_paths[i]); // read image fro the given filepath

            //checks if the input image contains data
            if(src.data == NULL) {
                printf("Unable to read image");
                exit(-1);
            }
            if(command == 1){
                processImageforRecognition(src);
            }else{
                processImageforRecognitionUsingDNNEmbeddings(src);
            }
        }

    }else if(command == 3 || command == 4 || command == 5){

        VideoCapture video("/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/3/utils/videos/Objects4.MOV");
        string featuresFileName;
        if(command == 3 || command == 5){
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/3/utils/featureFile/featureFile.csv";
        }else if(command == 4){
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/3/utils/featureFile/DNNEmbeddingsFeatureFile.csv";
        }

        if (!video.isOpened()) {
            cout<< "Cannot read the video file" << endl;
            return 0;
        }

        char windowName[256] = "objectRecognition";
        namedWindow(windowName, 1);
        std::vector<char*> objectNames;
        std::vector<std::vector<float>> objectFeatures;
        getImagesFeaturesData(featuresFileName, objectNames, objectFeatures);

        while (true) {
            video.read(src);
            if (src.empty()) {
                break;
            }
            if(command == 3){
                classifyImage(src, objectNames, objectFeatures);
            }else if(command == 4){
                classifyImageDNNEmbeddings(src, objectNames, objectFeatures);
            }else if(command == 5){
                cout<<"HERE"<<endl;
                classifyImageKNN(src, objectNames, objectFeatures, 3);
            }
            imshow(windowName, src);
            if (waitKey(30) >= 0) {
                break;
            }
        }
        video.release();
        destroyAllWindows();
    }

    printf("Terminating\n");

    return 0;
    
}
