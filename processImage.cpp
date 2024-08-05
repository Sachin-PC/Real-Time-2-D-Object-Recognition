/*
    Sachin Palahalli Chandrakumar
    Spring 2024
    Code for training and classifying objects
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "math.h"
#include <stack>

using namespace cv;
using namespace std;

// Defining the prototypes of methods used in this file
int append_image_data_csv( char *filename, char *image_filename, std::vector<float> &image_data, int reset_file );

/**
 * @brief applies the threshold to the given input image
 * This function take the input image and applies threshold 
 * @param[in] src input image
 * @param[in] dst output image after threshold applied
 * @return 0.
*/
int applyThreshold(cv::Mat &src, cv::Mat &dst){

    //checks if the input image contains data
    if(src.data == NULL) { 
        printf("Unable to read image ");
        exit(-1);
    }
    Mat greyScaleImg;
    cvtColor(src, greyScaleImg, COLOR_BGR2GRAY);
    
    int numRows = greyScaleImg.rows;
    int numCols = greyScaleImg.cols;
    int numChannels = greyScaleImg.channels();
    dst.create(numRows, numCols, greyScaleImg.type());

    int numBins = 256, pixelVal;
    int greyScaleHistogram[numBins];
    for(int i=0;i<numBins;i++){
            greyScaleHistogram[i] = 0;
    }

    for(int i=0;i<numRows;i++){
        for(int j=0;j<numCols;j++){
            pixelVal = greyScaleImg.at<uchar>(i, j);
            greyScaleHistogram[pixelVal] = greyScaleHistogram[pixelVal]+1;
        }
    }

    int midPixel = numBins/2;
    int peak1Max =0, peak1MaxIndex, peak2Max=0, peak2MaxIndex;
    for(int i=0;i<midPixel;i++){
        if(greyScaleHistogram[i] > peak1Max){
            peak1Max = greyScaleHistogram[i];
            peak1MaxIndex = i;
        }
    }
    for(int i=midPixel;i<numBins;i++){
        if(greyScaleHistogram[i] > peak2Max){
            peak2Max = greyScaleHistogram[i];
            peak2MaxIndex= i;
        }
    }

    int thresholdValueCount = peak1Max;
    int thresholdValue = peak1MaxIndex;
    for(int i=peak1MaxIndex;i<peak2MaxIndex;i++){
        if(greyScaleHistogram[i] < thresholdValueCount){
            thresholdValueCount = greyScaleHistogram[i];
            thresholdValue = i;
        }
    }

    for(int i=0;i<numRows;i++){
        for(int j=0;j<numCols;j++){
            pixelVal = greyScaleImg.at<uchar>(i, j);
            if(pixelVal >= thresholdValue){
                dst.at<uchar>(i, j) = (uchar)255;
            }else{
                dst.at<uchar>(i, j) = (uchar)0;
            }
        }
    }

    return 0;
}

/**
 * @brief cleans the image regions
 * This function take the input image and cleans the given image
 * @param[in] src input image
 * @param[in] dst output image after cleaning applied
 * @return 0.
*/
int cleanImage(cv::Mat &src, cv::Mat &dst){
    
    //Shrinking using 4*4 matrix
    Mat shrinkMatrix = getStructuringElement(MORPH_RECT, Size(4, 4));
    morphologyEx(src, dst, MORPH_ERODE, shrinkMatrix);

    // Growing using 8* matrix
    Mat growMatrix = getStructuringElement(MORPH_RECT, Size(8, 8));
    morphologyEx(dst, dst, MORPH_DILATE, growMatrix);

    return 0;
}

/**
 * @brief gets the rehgion statistics
 * @param[in] regionStats vector which will contain the region stats data
 * @param[in] left_col left_column pixel value
 * @param[in] right_col right_column pixel value
 * @param[in] top_row top row pixel value
 * @param[in] bottom_row bottom row pixel value
 * @param[in] total_pixels total pixle of the region
 * @return 0.
*/
int getReionStatsVector(std::vector<double> &regionStats,int left_col,int right_col,int top_row,int bottom_row,int total_pixels){
    
    int region_width = (right_col - left_col) + 1;
    int region_height = (bottom_row - top_row) + 1;
    int centroid_row = (top_row + (region_height/2));
    int centroid_col = (left_col + (region_width/2));
    regionStats.push_back(left_col);
    regionStats.push_back(right_col);
    regionStats.push_back(top_row);
    regionStats.push_back(bottom_row);
    regionStats.push_back(total_pixels);
    regionStats.push_back(region_width);
    regionStats.push_back(region_height);
    regionStats.push_back(centroid_row);
    regionStats.push_back(centroid_col);
    return 0;
}

/**
 * @brief segments the given image into multiple regions
 * @param[in] src input image
 * @param[in] regionMap region map containing region ids
 * @param[in] dst output image
 * @param[in] regionIds tcontaining regions ids of all regions
 * @param[in] regionsStatsMap vector which will contain the region stats data
 * @return 0.
*/
int segmentImage(cv::Mat &src, cv::Mat &regionMap, cv::Mat &dst, std::vector<int> &regionIds, std::unordered_map<int, std::vector<double>> &regionsStatsMap){

    //checks if the input image contains data
    if(src.data == NULL) { 
        printf("Unable to read image ");
        exit(-1);
    }
    
    int numRows = src.rows;
    int numCols = src.cols;
    int numChannels = src.channels();
    regionMap.create(numRows, numCols, src.type());
    dst.create(numRows, numCols, src.type());
    int thresholdRegionSize = 1000;
    int majorRejions = 0;
    //initialize region map with 0's
    for(int i=0;i<numRows;i++){
        for(int j=0;j<numCols;j++){
            regionMap.at<uchar>(i, j) = 0;
        }
    }

    int cur_regionId = 0, row_index=0, col_index=0;
    int left_col, right_col, top_row, bottom_row, total_pixels;
    bool regionFound = false;
    std::pair<int, int> pixelIndex;
    // std::unordered_map<int, std::vector<int>> regionsStatsMap;
    //define regions of pixels
    for(int i=0;i<numRows;i++){
        for(int j=0;j<numCols;j++){
            if((int)src.at<uchar>(i, j) == 0 && (int)regionMap.at<uchar>(i, j) == 0){
                total_pixels = 0;
                left_col = j;
                right_col =j;
                top_row = i;
                bottom_row = i;
                cur_regionId++;
                regionMap.at<uchar>(i, j) = cur_regionId;
                // cout<<"regionMap.at<uchar>(i, j) = "<<(int)regionMap.at<uchar>(i, j)<<endl;
                std::stack<std::pair<int, int>> regionPixels;
                regionPixels.push(std::make_pair(i,j));
                total_pixels++;
                cout<<"i = "<<i<<"j = "<<j<<endl;
                while (!regionPixels.empty()) {
                    
                    pixelIndex = regionPixels.top();
                    regionPixels.pop();
                    row_index = pixelIndex.first;
                    col_index = pixelIndex.second;
                    // if(total_pixels == 1){
                    //     cout<<"row_index = "<<row_index<<" col_index = "<<col_index<<endl;
                    // }
                    //setting the top,bottom row and left, right col of the region
                    if(row_index < top_row){
                        top_row = row_index;
                    }else if(row_index > bottom_row){
                        bottom_row = row_index;
                    }
                    if(col_index < left_col){
                        left_col = col_index;
                    }else if(col_index > right_col){
                        right_col = col_index;
                    }
                    for(int p=row_index-1; p<row_index+2; p++){
                        for(int q=col_index-1; q< col_index+2; q++){
                            if(!(p == row_index && q == col_index) && p >=0 && p < numRows && q >=0 && q < numCols ){
                                if((int)src.at<uchar>(p, q) == 0 && (int)regionMap.at<uchar>(p, q) == 0){
                                    // if(total_pixels < 10){
                                    //     cout<<"inside p = "<<p<<" q = "<<q<<"  src = "<<(int)src.at<uchar>(p, q)<<"  regionMap = "<<(int)regionMap.at<uchar>(p, q)<<endl;
                                    // }
                                    regionMap.at<uchar>(p, q) = cur_regionId;
                                    regionPixels.push(std::make_pair(p,q));
                                    total_pixels++;
                                }
                            }
                        }
                    }
                }

                //the region is found. Now add all the stats
                // std::vector<int> regionStats;
                // getReionStatsVector(regionStats,left_col,right_col,top_row,bottom_row,total_pixels);
                // regionsStatsMap[cur_regionId] = regionStats;
                cout<<"total_pixels = "<<total_pixels<<endl;
                std::vector<double> regionStats;
                getReionStatsVector(regionStats,left_col,right_col,top_row,bottom_row,total_pixels);
                regionsStatsMap[cur_regionId] = regionStats;
                if(total_pixels > thresholdRegionSize){
                    majorRejions++;
                    regionIds.push_back(cur_regionId);
                }
            }
        }
    }
    cout<<"Total number of regions = "<<cur_regionId<<"  Major regions == "<<majorRejions<<endl;
    // for(int i=0;i<regionIds.size();i++){
    //     std::vector<int> regionStats = regionsStatsMap[regionIds[i]];
    //     cout<<"------------"<<endl;
    //     cout<<"Region: "<<regionIds[i]<<endl;
    //     cout<<"left_col = "<<regionStats[0]<<endl;
    //     cout<<"right_col = "<<regionStats[1]<<endl;
    //     cout<<"top_row = "<<regionStats[2]<<endl;
    //     cout<<"bottom_row = "<<regionStats[3]<<endl;
    //     cout<<"total_pixels = "<<regionStats[4]<<endl;
    //     cout<<"region_width = "<<regionStats[5]<<endl;
    //     cout<<"region_height = "<<regionStats[6]<<endl;
    //     cout<<"centroid_row = "<<regionStats[7]<<endl;
    //     cout<<"centroid_col = "<<regionStats[8]<<endl;
    //     cout<<"------------"<<endl;
    // }
    for(int index=0;index<regionIds.size();index++){
        std::vector<double> regionStats = regionsStatsMap[regionIds[index]];
        int row = regionStats[2];
        int h = regionStats[6];
        int col = regionStats[0];
        int w = regionStats[5];
        int regionId = regionIds[index];
        double regionFilled=0;
        for(int i=row; i<row+h; i++){
            for(int j=col; j <col+w; j++){
                if(regionMap.at<uchar>(i, j) == regionId){
                    regionFilled++;
                }
            }
        }
        double regionFilledRatio = regionFilled/(h*w);
        double heightWidthRatio  = (double)h/(w+h);
        // cout<<"regionFilledRatio = "<<regionFilledRatio<<endl;
        // cout<<"heightWidthRatio = "<<heightWidthRatio<<endl;
        regionsStatsMap[regionIds[index]].push_back(regionFilledRatio);
        regionsStatsMap[regionIds[index]].push_back(heightWidthRatio);
    }

    int count=0;
    // making the region white and backgrond black
    int colorVal = 255;
    std::unordered_map<int, int> regionColorIds;
    for(int i=0; i< regionIds.size();i++){
        regionColorIds[regionIds[i]] = colorVal;
        if(colorVal > 55){
            colorVal = colorVal - 50;
        }
    }
    // cv::Mat regionDisplay;
    // regionDisplay.create(numRows, numCols, src.type());
    for(int i=0;i<numRows;i++){
        for(int j=0;j<numCols;j++){
            if((int)regionMap.at<uchar>(i, j) == 0 || regionsStatsMap[(int)regionMap.at<uchar>(i, j)][4] < thresholdRegionSize){
                // count++;
                dst.at<uchar>(i,j) = 0;
                // regionDisplay.at<uchar>(i,j) = 0;
            }else{
                dst.at<uchar>(i,j) = 255;
                // cout<<"regionColorIds[(int)regionMap.at<uchar>(i, j)] = "<<regionColorIds[(int)regionMap.at<uchar>(i, j)]<<endl;
                // regionDisplay.at<uchar>(i,j) = regionColorIds[(int)regionMap.at<uchar>(i, j)];
            }
        }
    }
    
    // char windowName[256] = "segmentedRegions";
    // namedWindow(windowName, 1);
    // imshow(windowName, regionDisplay);
    // while(true){
    //     char ch = waitKey(0);
    //     if(ch == 'q'){
    //         destroyWindow(windowName);
    //         break;
    //     }
    // }
    // cout<<"count = "<<count<<endl;

    // cout<<"Total Pixels of image = "<<numRows*numCols<<endl;

    return majorRejions;
}

/**
 * @brief sgets the region with input region id
 * @param[in] regionMap region map containing region ids
 * @param[in] regionImage output image containing required region
 * @param[in] regionId regions id of the region
 * @return 0.
*/
int getSingleRegionImage(cv::Mat &regionMap, cv::Mat &regionImage, int regionId){
    
    int numRows = regionImage.rows;
    int numCols = regionImage.cols;
    
    for(int i=0;i<numRows;i++){
        for(int j=0;j<numCols;j++){
            if(i == 0 || j == 0){
                regionImage.at<uchar>(i,j) = 0;
            }
            else if((int)regionMap.at<uchar>(i, j) == regionId){
                // count++;
                regionImage.at<uchar>(i,j) = 255;
            }else{
                regionImage.at<uchar>(i,j) = 0;
            }
        }
    }

    return 0;
}

/**
 * @brief gets the region moments
 * @param[in] regionImage region image
 * @param[in] regionMoments moments of the region image
 * @return 0.
*/
int getRegionMoments(cv::Mat &regionImage, cv::Mat regionMoments){

    int numRows = regionImage.rows;
    int numCols = regionImage.cols;
    double momentsVal = 0.0;
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            momentsVal = 0.0;
            for(int x=0; x<numRows; x++){
                for(int y=0; y<numCols; y++){
                    if((int)regionImage.at<uchar>(x,y) != 0){
                        momentsVal += std::pow(x, i)*std::pow(y, j);
                    }
                }
            }
            regionMoments.at<double>(i,j) = momentsVal;
            cout<<"moments("<<i<<", "<<j<<") = "<<momentsVal<<endl;
        }
    }

    int c_y = regionMoments.at<double>(1,0)/regionMoments.at<double>(0,0);
    int c_x = regionMoments.at<double>(0,1)/regionMoments.at<double>(0,0);
    cout<<"centroid x = "<<regionMoments.at<double>(1,0)/regionMoments.at<double>(0,0)<<endl;
    cout<<"centroid y = "<<regionMoments.at<double>(0,1)/regionMoments.at<double>(0,0)<<endl;
    
    // for(int i=c_x-3;i<c_x+3;i++){
    //     for(int j=c_y-3;j<c_y+3;j++){
    //         regionImage.at<uchar>(i,j) = (int)125;
    //     }
    // }

    return 0;
}

/**
 * @brief computes the features of the regions present in the image
 * @param[in] src input image
 * @param[in] regionMap region map containing region ids
 * @param[in] allRegionImage image containing all regions
 * @param[in] totalRegions count of regions
 * @return 0.
*/
int createFeatures(cv::Mat &regionMap, cv::Mat &allRegionImage, int totalRegions){

    int numRows = allRegionImage.rows;
    int numCols = allRegionImage.cols;
    int regionId=0;
    for(int regionId=1;regionId<=totalRegions;regionId++){
        cv::Mat regionImage;
        cv::Mat regionMoments(2, 2, CV_64F);
        // 0th row and 0th column are dummy because we should not consider row and column index as 0 in calculating moments
        regionImage.create(numRows+1, numCols+1, allRegionImage.type());
        getSingleRegionImage(regionMap, regionImage, regionId);
        getRegionMoments(regionImage, regionMoments);
    }
    return 0;
}

/**
 * @brief rotates the image along x axis
 * @param[in] regionImage image containing region
 * @param[in] rotatedRegion output rotated region
 * @return 0.
*/
int getRotatedImage(cv::Mat &regionImage, cv::Mat &rotatedRegion){

    cv::Moments moments = cv::moments(regionImage, true);

    //image orientation angle
    double theta = 0.5*std::atan2(2*moments.mu11, moments.mu20 - moments.mu02);

    //rotate the image to align with x axis
    cv::Point2f centroid(moments.m10/moments.m00, moments.m01/moments.m00);
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(centroid, theta*180/CV_PI, 1.0);
    cv::Size regionSize(2*regionImage.rows, 2*regionImage.cols);
    cv::warpAffine(regionImage, rotatedRegion, rotationMatrix, regionSize);

    int left_col = rotatedRegion.cols;
    int right_col =0;
    int top_row = rotatedRegion.rows;
    int bottom_row = 0;
    for(int row_index=0; row_index<rotatedRegion.rows;row_index++ ){
        for(int col_index= 0; col_index < rotatedRegion.cols; col_index++){
            if(rotatedRegion.at<uchar>(row_index, col_index) == 255){
                if(row_index < top_row){
                    top_row = row_index;
                }else if(row_index > bottom_row){
                    bottom_row = row_index;
                }
                if(col_index < left_col){
                    left_col = col_index;
                }else if(col_index > right_col){
                    right_col = col_index;
                }
            }
        }
    }

    // for(int row_index=0; row_index<rotatedRegion.rows;row_index++ ){
    //     for(int col_index= 0; col_index < rotatedRegion.cols; col_index++){
    //         if(col_index < left_col && rotatedRegion.at<uchar>(row_index, col_index) != 0){
    //             cout<<"("<<row_index<<", "<<col_index<<") = "<<(int)rotatedRegion.at<uchar>(row_index, col_index)<<endl;
    //         }
    //     }
    // }

    // for(int i = top_row - 10; i< top_row+10;i++){
    //     for(int j=left_col-10; j< left_col+10;j++){
    //         rotatedRegion.at<uchar>(i, j) = 128;
    //     }
    // }

    int width = right_col - left_col + 1;
    int height = bottom_row - top_row + 1;
    // cout<<"height = "<<height<<endl;
    // cout<<" width = "<<width<<endl;
    cv::Rect boundingBox(left_col, top_row, width, height);
    cv::rectangle(rotatedRegion, boundingBox, cv::Scalar(255), 2);

    return 0;
}


/**
 * @brief computes feature vector for the given image region
 * @param[in] regionImage image containing region
 * @param[in] featureVector feature vector containing huMoments of the region input
 * @return 0.
*/
int getHuMomentsFeatureVector(cv::Mat &regionImage, std::vector<float> &featureVector){
        cv::Moments moments = cv::moments(regionImage, true);
        cv::Mat huMoments;
        cv::HuMoments(moments, huMoments);
        for (int i=0; i<7; i++) {
            featureVector.push_back(huMoments.at<double>(i, 0));
        }
        return 0;
}

/**
 * @brief processes the image for reciognition
 * @param[in] src input region
 * @return 0.
*/
int processImageforRecognition(cv::Mat &src){

    string featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/3/utils/featureFile/featureFile.csv";
    char windowName[256] = "objectRecognition";

    cv::Mat thresholdImage, cleanedImg, regionMap, labeledImage, allRegionImage;
    applyThreshold(src, thresholdImage);
    namedWindow(windowName, 1);
    imshow(windowName, thresholdImage);
    while(true){
        char ch = waitKey(0);
        if(ch == 'q'){
            destroyWindow(windowName);
            break;
        }
    }
    // cv::threshold(src, thresholdImage, 128, 255, cv::THRESH_BINARY);
    cleanImage(thresholdImage, cleanedImg);

    // char windowName[256] = "test";
    // namedWindow(windowName, 1);
    imshow(windowName, cleanedImg);
    while(true){
        char ch = waitKey(0);
        if(ch == 'q'){
            destroyWindow(windowName);
            break;
        }
    }

    std::vector<int> regionIds;
    std::unordered_map<int, std::vector<double>> regionsStatsMap;
    int totalRegions = segmentImage(cleanedImg,regionMap, allRegionImage, regionIds, regionsStatsMap);

    cout<<"Total Regions = "<<totalRegions<<endl;
    int numRows = allRegionImage.rows;
    int numCols = allRegionImage.cols;
    int regionId=0;
    std::string objectLabel; 
    int reset_file = 0;
    for(int i=0;i<totalRegions;i++){
        cout<<"Region id = "<<regionIds[i]<<endl;
        cv::Mat regionImage;
        // cv::Mat regionMoments(2, 2, CV_64F);
        // 0th row and 0th column are dummy because we should not consider row and column index as 0 in calculating moments
        regionImage.create(numRows+1, numCols+1, allRegionImage.type());
        getSingleRegionImage(regionMap, regionImage, regionIds[i]);
        cv::Mat rotatedRegion;
        getRotatedImage(regionImage, rotatedRegion);
        namedWindow(windowName, 1);
        imshow(windowName, rotatedRegion);
        while(true){
            char ch = waitKey(0);
            if(ch == 'q'){
                destroyWindow(windowName);
                break;
            }
        }   

        cout<<"Please enter the object present in the image"<<endl;
        std::cin>>objectLabel;
        std::vector<float> featureVector;
        getHuMomentsFeatureVector(regionImage, featureVector);
        featureVector.push_back(regionsStatsMap[regionIds[i]][9]);
        featureVector.push_back(regionsStatsMap[regionIds[i]][10]);
        append_image_data_csv(&featuresFileName[0], &objectLabel[0],  featureVector, reset_file);
        // imshow(imgFileName, rotatedRegion);
    }
    return 0;
}

/**
 * @brief calculates the dnn embedding of the inputimage
 * @param[in] src input region
 * @param[in] embeddingFeatureVector embedding feature vector
 * @param[in] new deep neural network model
 * @return 0.
*/
int getDNNEmbeddings( cv::Mat &src, std::vector<float> &embeddingFeatureVector, cv::dnn::Net &net ) {
    const int ORNet_size = 128;
    cv::Mat blob;
    cv::Mat embedding;
        
    cv::dnn::blobFromImage( src, // input image
                blob, // output array
                (1.0/255.0) / 0.5, // scale factor
                cv::Size( ORNet_size, ORNet_size ), // resize the image to this
                128,   // subtract mean prior to scaling
                false, // input is a single channel image
                true,  // center crop after scaling short side to size
                CV_32F ); // output depth/type

    net.setInput( blob );
    embedding = net.forward( "onnx_node!/fc1/Gemm" );

    for(int i=0; i<embedding.rows;i++){
        for(int j=0; j< embedding.cols; j++){
            embeddingFeatureVector.push_back(embedding.at<float>(i,j));
        }
    }

    return(0);
}

/**
 * @brief processes the image for reciognition
 * @param[in] src input region
 * @return 0.
*/
int processImageforRecognitionUsingDNNEmbeddings(cv::Mat &src){

    char mod_filename[256] = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/3/utils/dnnmodel/1.onnx";

    // read the network
    cv::dnn::Net net = cv::dnn::readNet( mod_filename );
    printf("Network read successfully\n");

    string featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/3/utils/featureFile/DNNEmbeddingsFeatureFile.csv";
    char windowName[256] = "objectRecognition";

    cv::Mat thresholdImage, cleanedImg, regionMap, labeledImage, allRegionImage;
    applyThreshold(src, thresholdImage);
    namedWindow(windowName, 1);
    imshow(windowName, thresholdImage);
    while(true){
        char ch = waitKey(0);
        if(ch == 'q'){
            destroyWindow(windowName);
            break;
        }
    }
    // cv::threshold(src, thresholdImage, 128, 255, cv::THRESH_BINARY);
    cleanImage(thresholdImage, cleanedImg);

    // char windowName[256] = "test";
    // namedWindow(windowName, 1);
    imshow(windowName, cleanedImg);
    while(true){
        char ch = waitKey(0);
        if(ch == 'q'){
            destroyWindow(windowName);
            break;
        }
    }

    std::vector<int> regionIds;
    std::unordered_map<int, std::vector<double>> regionsStatsMap;
    int totalRegions = segmentImage(cleanedImg,regionMap, allRegionImage, regionIds, regionsStatsMap);

    cout<<"Total Regions = "<<totalRegions<<endl;
    int numRows = allRegionImage.rows;
    int numCols = allRegionImage.cols;
    int regionId=0;
    std::string objectLabel; 
    int reset_file = 0;
    for(int i=0;i<totalRegions;i++){
        cout<<"Region id = "<<regionIds[i]<<endl;
        cv::Mat regionImage;
        // cv::Mat regionMoments(2, 2, CV_64F);
        // 0th row and 0th column are dummy because we should not consider row and column index as 0 in calculating moments
        regionImage.create(numRows+1, numCols+1, allRegionImage.type());
        getSingleRegionImage(regionMap, regionImage, regionIds[i]);
        // cv::Mat rotatedRegion;
        // getRotatedImage(regionImage, rotatedRegion);
        // namedWindow(windowName, 1);
        imshow(windowName, regionImage);
        while(true){
            char ch = waitKey(0);
            if(ch == 'q'){
                destroyWindow(windowName);
                break;
            }
        }   

        cout<<"Please enter the object present in the image"<<endl;
        std::cin>>objectLabel;
        std::vector<float> DNNEmbeddingfeatureVector;
        getDNNEmbeddings( regionImage, DNNEmbeddingfeatureVector, net);
        append_image_data_csv(&featuresFileName[0], &objectLabel[0],  DNNEmbeddingfeatureVector, reset_file);
        // imshow(imgFileName, rotatedRegion);
    }
    return 0;
}


/**
 * @brief gets the image regions features vectors
 * @param[in] src input image
 * @param[in] allRegionFeatures features of all regions
 * @param[in] regionsStatsMap vector which will contain the region stats data
 * @param[in] regionIds tcontaining regions ids of all regions
 * @return 0.
*/
int getImageRegionsFeatureVectors(cv::Mat &src, std::vector<std::vector<float>> &allRegionFeatures, std::unordered_map<int, std::vector<double>> &regionsStatsMap, std::vector<int> &regionIds){


    cv::Mat thresholdImage, cleanedImg, regionMap, labeledImage, allRegionImage;
    applyThreshold(src, thresholdImage);
    cleanImage(thresholdImage, cleanedImg);
    
    int totalRegions = segmentImage(cleanedImg,regionMap, allRegionImage, regionIds, regionsStatsMap);

    int numRows = allRegionImage.rows;
    int numCols = allRegionImage.cols;
    int regionId=0;
    std::string objectLabel; 
    int reset_file = 0;
    for(int i=0;i<totalRegions;i++){
        cv::Mat regionImage;
        // 0th row and 0th column are dummy because we should not consider row and column index as 0 in calculating moments
        regionImage.create(numRows+1, numCols+1, allRegionImage.type());
        getSingleRegionImage(regionMap, regionImage, regionIds[i]);
        cv::Mat rotatedRegion;
        getRotatedImage(regionImage, rotatedRegion);

        std::vector<float> featureVector;
        getHuMomentsFeatureVector(regionImage, featureVector);
        allRegionFeatures.push_back(featureVector);
    }
    
    return totalRegions;
}


/**
 * @brief gets the image regions features vectors
 * @param[in] src input image
 * @param[in] allRegionFeatures features of all regions
 * @param[in] regionsStatsMap vector which will contain the region stats data
 * @param[in] regionIds tcontaining regions ids of all regions
 * @param[in] net neural network model
 * @return 0.
*/
int getImageRegionsDNNEmbeddingsFeatureVectors(cv::Mat &src, std::vector<std::vector<float>> &allRegionFeatures, std::unordered_map<int, std::vector<double>> &regionsStatsMap, std::vector<int> &regionIds, cv::dnn::Net net){


    cv::Mat thresholdImage, cleanedImg, regionMap, labeledImage, allRegionImage;
    applyThreshold(src, thresholdImage);
    cleanImage(thresholdImage, cleanedImg);
    
    int totalRegions = segmentImage(cleanedImg,regionMap, allRegionImage, regionIds, regionsStatsMap);

    int numRows = allRegionImage.rows;
    int numCols = allRegionImage.cols;
    int regionId=0;
    std::string objectLabel; 
    int reset_file = 0;
    for(int i=0;i<totalRegions;i++){
        cv::Mat regionImage;
        // 0th row and 0th column are dummy because we should not consider row and column index as 0 in calculating moments
        regionImage.create(numRows+1, numCols+1, allRegionImage.type());
        getSingleRegionImage(regionMap, regionImage, regionIds[i]);
        // cv::Mat rotatedRegion;
        // getRotatedImage(regionImage, rotatedRegion);

        // std::vector<float> featureVector;
        // getHuMomentsFeatureVector(regionImage, featureVector);
        std::vector<float> DNNEmbeddingfeatureVector;
        getDNNEmbeddings( regionImage, DNNEmbeddingfeatureVector, net);
        allRegionFeatures.push_back(DNNEmbeddingfeatureVector);
    }
    
    return totalRegions;
}

/**
 * @brief gets the Features standard deviations to scale the features
 * @param[in] trainedFetures feature values
 * @param[in] featuresSTDs vector contai9nig standard deviations of features
 * @return 0.
*/
int getFeaturesSTDs(std::vector<std::vector<float>> trainedFetures, std::vector<double> &featuresSTDs){

    int noOfObjects = trainedFetures.size();
    int noOfFeatures = trainedFetures[0].size();
    double mean_value=0, variance_val = 0, std_val=0;
    for(int i=0; i<noOfFeatures ;i++){
        std::vector<float> singleFeatureValues;
        for(int j=0; j<noOfObjects;j++){
            singleFeatureValues.push_back(trainedFetures[j][i]);
        }
        mean_value = 0;
        for(int j=0;j< singleFeatureValues.size(); j++){
            mean_value += singleFeatureValues[j];
        }
        mean_value = mean_value / noOfObjects;
        variance_val = 0;
        for(int j=0;j< singleFeatureValues.size(); j++){
            variance_val += (singleFeatureValues[j] - mean_value)*(singleFeatureValues[j] - mean_value);
        }
        variance_val = variance_val/noOfObjects;
        std_val = std::sqrt(variance_val);
        featuresSTDs.push_back(std_val);
    }
    return 0;
}

/**
 * @brief gets the eucledian distance between two vectors
 * @param[in] allRegionFeatures features of all regions
 * @param[in] trainedObjectFeatureVector trained object feature vector
 * @param[in] distanceScores containing distance scores of objects
 * @param[in] featuresSTDs vector contai9nig standard deviations of features
 * @return 0.
*/
int getEucledianDistanceScore(std::vector<std::vector<float>> allRegionFeatures, std::vector<float> &trainedObjectFeatureVector, std::vector<double> &distanceScores, std::vector<double> featuresStds){

    std::vector<float> regionFeatureVector;
    double distance;
    // cout<<"allRegionFeatures size = "<<allRegionFeatures.size()<<endl;
    for(int i=0;i<allRegionFeatures.size(); i++){
        regionFeatureVector = allRegionFeatures[i];
        // for(int j=0;j<regionFeatureVector.size();j++){
        //     cout<<"regionFeatureVector["<<j<<"] = "<<regionFeatureVector[j]<<endl;
        // }
        distance = 0;
        for(int j=0; j<regionFeatureVector.size(); j++){
            distance += (trainedObjectFeatureVector[j] - regionFeatureVector[j])*(trainedObjectFeatureVector[j] - regionFeatureVector[j])/featuresStds[j];
        }
        distanceScores.push_back(distance);
    }
    // cout<<"returing 0";
    return 0;
}


/**
 * @brief gets the best score
 * @param[in] regionsDistanceVals regions distance values
 * @param[in] bestIndexes best indexes computed
 * @return 0.
*/
int getBestScoreIndex(std::vector<std::vector<float>> &regionsDistanceVals, std::vector<int> &bestIndexes){
    
    std::vector<float> distanceVals;
    double min_val=0;
    int min_val_index=0;
    for(int i=0;i<regionsDistanceVals.size();i++){
        distanceVals = regionsDistanceVals[i];
        min_val = distanceVals[0];
        min_val_index = 0;
        for(int j=1; j< distanceVals.size(); j++){
            if(distanceVals[j] < min_val){
                min_val = distanceVals[j];
                min_val_index = j;
            }
            // cout<<"distanceVals["<<j<<"] = "<<distanceVals[j]<<"  min_vcal = "<<min_val<<"  min_val_index = "<<min_val_index<<endl;
        }
        cout<<"\n\nBest score = "<<min_val<<endl;
        bestIndexes.push_back(min_val_index);
    }

    return 0;
}

/**
 * @brief draws the bounding box
 * @param[in] inputImage input image
 * @param[in] regionsStatsMap region stats map
 * @param[in] totalRegions total regions
 * @param[in] regionIds region ids
 * @param[in] regionsNamePlaceHolder place holder names of regions
 * @return 0.
*/
int drawBoundingboxesForRegions(cv::Mat &inputImage, std::unordered_map<int, std::vector<double>> &regionsStatsMap, int totalRegions, std::vector<int> &regionIds, std::vector<cv::Point> &regionsNamePlaceHolder){
   
   int width, height, left_col, top_row;
    for(int i=0;i<totalRegions;i++){
        std::vector<double> regionStats = regionsStatsMap[regionIds[i]];
        left_col = regionStats[0];
        top_row = regionStats[2];
        width = regionStats[5];
        height = regionStats[6];

        cv::Rect boundingBox(left_col, top_row, width, height);
        cv::rectangle(inputImage, boundingBox, cv::Scalar(255), 2);
        cv::Point regionNamePosition(left_col, top_row - 10);
        regionsNamePlaceHolder.push_back(regionNamePosition);
    }

    return 0; 
}


/**
 * @brief given the feature vector and target image, finds the top images matching the target image
 * @param[in] targetImageName The target image name
 * @param[in] fileNames the image names
 * @param[in] data feature vectore of images data
 * @param[in] topN ftop n matches required
 * @param[in] getDistanceScore function to calculate the distance between the feature vectors
 * @param[in] isAscendingRank fis rank considerd in ascending order or descending order.
 * @param[in] bestMatchIndexes contains image names of the best matched
 * @return 0.
*/
int classifyImage(cv::Mat &inputImage, std::vector<char*> objectNames, std::vector<std::vector<float>> trainedObjectFeatures){



    int totalFiles = objectNames.size();
    // for(int i=0; i<totalFiles; i++){
    //     cout<<objectNames[i];
    // }
    std::vector<float> inputImageVector;
    std::vector<float> featureVector;
    std::vector<std::vector<float>> allRegionFeatures;
    std::unordered_map<int, std::vector<double>> regionsStatsMap;
    std::vector<int> regionIds;
    std::vector<cv::Point> regionsNamePlaceHolder;
    int totalRegions = getImageRegionsFeatureVectors(inputImage, allRegionFeatures, regionsStatsMap, regionIds);
    drawBoundingboxesForRegions(inputImage,regionsStatsMap,totalRegions,regionIds, regionsNamePlaceHolder);
 
    std::vector<double> distanceVector;
    std::vector<float> trainedObjectFeatureVector;
    std::vector<float> regionFeatureVector;
    std::vector<std::vector<float>> regionsDistanceVals;

    //initialize a 2d vector(because multiple regions are considered) where each vector will store the distance of it from the trained images
    for(int i=0; i < allRegionFeatures.size(); i++){
        std::vector<float> distanceVals;
        regionsDistanceVals.push_back(distanceVals);
        // for(int j=0; j<allRegionFeatures[i].size();j++){
        //     cout<<" feature["<<j<<"] = "<<allRegionFeatures[i][j]<<endl;
        // }
    }

    //get feature standard deviation values
    std::vector<double> featuresSTDs;
    getFeaturesSTDs(trainedObjectFeatures, featuresSTDs);

    //get scores for all regions
    // cout<<"totalFiles = "<<totalFiles<<endl;
    // for(int i=0;i<featuresSTDs.size();i++){
    //     cout<<"std feature"<<i<<" = "<<featuresSTDs[i]<<endl;
    //     // featuresSTDs[i] = 1;
    // }
    for(int i=0;i<totalFiles;i++){
        // cout<<"i = "<<i<<endl;
        trainedObjectFeatureVector = trainedObjectFeatures[i];
        std::vector<double> distanceScores; //stores distance of each region from the given trainde object
        getEucledianDistanceScore(allRegionFeatures, trainedObjectFeatureVector, distanceScores,featuresSTDs);
        // for(int j=0;j<distanceScores.size();j++){
        //     cout<<"distanceScores["<<j<<"] = "<<distanceScores[j]<<endl;
        // }
        for(int j=0; j < regionsDistanceVals.size(); j++){
            regionsDistanceVals[j].push_back(distanceScores[j]);
        }
    }

    
    //get the index of the image which is closest to each region
    std::vector<int> bestindexes;
    getBestScoreIndex(regionsDistanceVals, bestindexes);

    //get the names of the regions present at that index and poulate the text field
    for(int i=0; i< bestindexes.size(); i++){
        std::string classifiedRegionName = std::string(objectNames[bestindexes[i]]);
        // std::string classifiedRegionName = "xyz";
        cv::putText(inputImage, classifiedRegionName, regionsNamePlaceHolder[i], cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2);
    }

    // char windowName[256] = "test";
    // namedWindow(windowName, 1);
    // imshow(windowName, inputImage);
    // while(true){
    //     char ch = waitKey(0);
    //     if(ch == 'q'){
    //         destroyWindow(windowName);
    //         break;
    //     }
    // }

    return 0;
}


/**
 * @brief given the feature vector and target image, finds the top images matching the target image
 * @param[in] targetImageName The target image name
 * @param[in] fileNames the image names
 * @param[in] data feature vectore of images data
 * @param[in] topN ftop n matches required
 * @param[in] getDistanceScore function to calculate the distance between the feature vectors
 * @param[in] isAscendingRank fis rank considerd in ascending order or descending order.
 * @param[in] bestMatchIndexes contains image names of the best matched
 * @return 0.
*/
int classifyImageDNNEmbeddings(cv::Mat &inputImage, std::vector<char*> objectNames, std::vector<std::vector<float>> trainedObjectFeatures){


    char mod_filename[256] = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/3/utils/dnnmodel/1.onnx";

    // read the network
    cv::dnn::Net net = cv::dnn::readNet( mod_filename );
    printf("Network read successfully\n");

    int totalFiles = objectNames.size();
    for(int i=0; i<totalFiles; i++){
        cout<<objectNames[i];
    }
    std::vector<float> inputImageVector;
    std::vector<float> featureVector;
    std::vector<std::vector<float>> allRegionFeatures;
    std::unordered_map<int, std::vector<double>> regionsStatsMap;
    std::vector<int> regionIds;
    std::vector<cv::Point> regionsNamePlaceHolder;
    int totalRegions = getImageRegionsDNNEmbeddingsFeatureVectors(inputImage, allRegionFeatures, regionsStatsMap, regionIds, net);
    drawBoundingboxesForRegions(inputImage,regionsStatsMap,totalRegions,regionIds, regionsNamePlaceHolder);
 
    std::vector<double> distanceVector;
    std::vector<float> trainedObjectFeatureVector;
    std::vector<float> regionFeatureVector;
    std::vector<std::vector<float>> regionsDistanceVals;

    //initialize a 2d vector(because multiple regions are considered) where each vector will store the distance of it from the trained images
    for(int i=0; i < allRegionFeatures.size(); i++){
        std::vector<float> distanceVals;
        regionsDistanceVals.push_back(distanceVals);
        for(int j=0; j<allRegionFeatures[i].size();j++){
            cout<<" feature["<<j<<"] = "<<allRegionFeatures[i][j]<<endl;
        }
    }

    //get feature standard deviation values
    std::vector<double> featuresSTDs;
    getFeaturesSTDs(trainedObjectFeatures, featuresSTDs);

    //get scores for all regions
    cout<<"totalFiles = "<<totalFiles<<endl;
    for(int i=0;i<featuresSTDs.size();i++){
        cout<<"std feature"<<i<<" = "<<featuresSTDs[i]<<endl;
        // featuresSTDs[i] = 1;
    }
    for(int i=0;i<totalFiles;i++){
        cout<<"i = "<<i<<endl;
        trainedObjectFeatureVector = trainedObjectFeatures[i];
        std::vector<double> distanceScores; //stores distance of each region from the given trainde object
        getEucledianDistanceScore(allRegionFeatures, trainedObjectFeatureVector, distanceScores,featuresSTDs);
        for(int j=0;j<distanceScores.size();j++){
            cout<<"distanceScores["<<j<<"] = "<<distanceScores[j]<<endl;
        }
        for(int j=0; j < regionsDistanceVals.size(); j++){
            regionsDistanceVals[j].push_back(distanceScores[j]);
        }
    }

    cout<<"44444"<<endl;
    
    //get the index of the image which is closest to each region
    std::vector<int> bestindexes;
    getBestScoreIndex(regionsDistanceVals, bestindexes);

    //get the names of the regions present at that index and poulate the text field
    for(int i=0; i< bestindexes.size(); i++){
        std::string classifiedRegionName = std::string(objectNames[bestindexes[i]]);
        // std::string classifiedRegionName = "xyz";
        cv::putText(inputImage, classifiedRegionName, regionsNamePlaceHolder[i], cv::FONT_HERSHEY_SIMPLEX, 4, cv::Scalar(0, 0, 255), 2);
    }

    // char windowName[256] = "test";
    // namedWindow(windowName, 1);
    // imshow(windowName, inputImage);
    // while(true){
    //     char ch = waitKey(0);
    //     if(ch == 'q'){
    //         destroyWindow(windowName);
    //         break;
    //     }
    // }



    return 0;
}

/**
 * @brief retursn the best class detected
 * @param[in] objectNames the image names
 * @param[in] regionsDistanceVals region distance values computed
 * @param[in] predictedRegionClasses the results of the predicted class values
 * @return 0.
*/
int getBestClass(std::vector<char*> objectNames, std::vector<std::vector<float>> &regionsDistanceVals, int k, std::vector<string> &predictedRegionClasses){


    std::vector<float> distanceVals;
    for(int i=0;i<regionsDistanceVals.size();i++){
        // cout<<"Region "<<i<<endl;
        distanceVals = regionsDistanceVals[i];
        std::unordered_map<std::string, std::vector<double>> classDistanceScores;
        for(int j=1; j< distanceVals.size(); j++){
            classDistanceScores[string(objectNames[j])].push_back(distanceVals[j]);
        }
        std::string best_class;
        double min_sum=9999999999;
        for (auto j = classDistanceScores.begin(); j != classDistanceScores.end(); j++) {
            // cout<<"Object = "<< j->first<<endl;
            std::vector<double>  distances = j->second;
            if(distances.size() > k){
                std::sort(distances.begin(), distances.end());
                double distance_sum=0;
                for(int index=0;index<k;index++){
                    // cout<<"Distances["<<index<<"] = "<<distances[index]<<"\t";
                    distance_sum+= distances[index];
                }
                // cout<<"distance_sum = "<<distance_sum<<endl;
                if(distance_sum < min_sum){
                    min_sum = distance_sum;
                    best_class = j->first;
                }
                // cout<<"min_sum = "<<min_sum<<endl;
                // cout<<"best_class = "<<best_class<<endl;
            }
        }
        // cout<<"final best_class = "<<best_class<<endl;
        predictedRegionClasses.push_back(best_class);
        cout<<endl;
    }

    return 0;
    
}


/**
 * @brief given the feature vector and target image, finds the top images matching the target image
 * @param[in] targetImageName The target image name
 * @param[in] fileNames the image names
 * @param[in] data feature vectore of images data
 * @param[in] topN ftop n matches required
 * @param[in] getDistanceScore function to calculate the distance between the feature vectors
 * @param[in] isAscendingRank fis rank considerd in ascending order or descending order.
 * @param[in] bestMatchIndexes contains image names of the best matched
 * @return 0.
*/
int classifyImageKNN(cv::Mat &inputImage, std::vector<char*> objectNames, std::vector<std::vector<float>> trainedObjectFeatures, int k_neighbours){



    int totalFiles = objectNames.size();
    // for(int i=0; i<totalFiles; i++){
    //     cout<<objectNames[i];
    // }
    std::vector<float> inputImageVector;
    std::vector<float> featureVector;
    std::vector<std::vector<float>> allRegionFeatures;
    std::unordered_map<int, std::vector<double>> regionsStatsMap;
    std::vector<int> regionIds;
    std::vector<cv::Point> regionsNamePlaceHolder;
    int totalRegions = getImageRegionsFeatureVectors(inputImage, allRegionFeatures, regionsStatsMap, regionIds);
    drawBoundingboxesForRegions(inputImage,regionsStatsMap,totalRegions,regionIds, regionsNamePlaceHolder);
 
    std::vector<double> distanceVector;
    std::vector<float> trainedObjectFeatureVector;
    std::vector<float> regionFeatureVector;
    std::vector<std::vector<float>> regionsDistanceVals;

    //initialize a 2d vector(because multiple regions are considered) where each vector will store the distance of it from the trained images
    for(int i=0; i < allRegionFeatures.size(); i++){
        std::vector<float> distanceVals;
        regionsDistanceVals.push_back(distanceVals);
        // for(int j=0; j<allRegionFeatures[i].size();j++){
        //     cout<<" feature["<<j<<"] = "<<allRegionFeatures[i][j]<<endl;
        // }
    }

    //get feature standard deviation values
    std::vector<double> featuresSTDs;
    getFeaturesSTDs(trainedObjectFeatures, featuresSTDs);

    //get scores for all regions
    // cout<<"totalFiles = "<<totalFiles<<endl;
    // for(int i=0;i<featuresSTDs.size();i++){
    //     cout<<"std feature"<<i<<" = "<<featuresSTDs[i]<<endl;
    //     // featuresSTDs[i] = 1;
    // }
    for(int i=0;i<totalFiles;i++){
        // cout<<"i = "<<i<<endl;
        trainedObjectFeatureVector = trainedObjectFeatures[i];
        std::vector<double> distanceScores; //stores distance of each region from the given trainde object
        getEucledianDistanceScore(allRegionFeatures, trainedObjectFeatureVector, distanceScores,featuresSTDs);
        // for(int j=0;j<distanceScores.size();j++){
        //     cout<<"distanceScores["<<j<<"] = "<<distanceScores[j]<<endl;
        // }
        for(int j=0; j < regionsDistanceVals.size(); j++){
            regionsDistanceVals[j].push_back(distanceScores[j]);
        }
    }

    
    // get the index of the image which is closest to each region
    // std::vector<int> bestindexes;
    // getBestScoreIndex(regionsDistanceVals, bestindexes);

    std::vector<string> predictedRegionClasses;
    getBestClass(objectNames, regionsDistanceVals, k_neighbours,predictedRegionClasses );

    //get the names of the regions present at that index and poulate the text field
    for(int i=0; i< predictedRegionClasses.size(); i++){
        // std::string classifiedRegionName = std::string(objectNames[bestindexes[i]]);
        // cout<<"predictedRegionClasses = "<<predictedRegionClasses[i]<<endl;
        std::string classifiedRegionName = predictedRegionClasses[i];
        cv::putText(inputImage, classifiedRegionName, regionsNamePlaceHolder[i], cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2);
    }

    // char windowName[256] = "test";
    // namedWindow(windowName, 1);
    // imshow(windowName, inputImage);
    // while(true){
    //     char ch = waitKey(0);
    //     if(ch == 'q'){
    //         destroyWindow(windowName);
    //         break;
    //     }
    // }

    return 0;
}
