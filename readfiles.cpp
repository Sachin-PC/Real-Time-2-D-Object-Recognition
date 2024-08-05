/*
  Code Provided Bruce A. Maxwell
  S21
  
  Sample code to identify image fils in a directory
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <iostream>
#include <vector>
#include "math.h"

using namespace std;
/*
  Given a directory on the command line, scans through the directory for image files.

  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */

int get_image_paths( std::vector<string> &image_paths){

  char dirname[256] = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/3/utils/TrainingImages";
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  int i;


  // get the directory path
  // strcpy(dirname, argv[1]);
  printf("Processing directory %s\n", dirname );

  // open the directory
  dirp = opendir( dirname );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  // loop over all the files in the image file listing
  while( (dp = readdir(dirp)) != NULL ) {

    // check if the file is an image
    if( strstr(dp->d_name, ".jpg") ||
	strstr(dp->d_name, ".png") ||
	strstr(dp->d_name, ".ppm") ||
	strstr(dp->d_name, ".tif") ||
  strstr(dp->d_name, ".heic") ||
  strstr(dp->d_name, ".jpeg")) {

      // printf("processing image file: %s\n", dp->d_name);

      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);
      std::string xyz(buffer);
      image_paths.push_back(xyz);
      // std::cout << "String: " << xyz << std::endl;
      // printf("full path name: %s\n", xyz);

    }
  }
  
  printf("Returning\n");

  return(0);

}



int test_main(int argc, char *argv[]) {
  char dirname[256] = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/olympus";
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  int i;

  std::vector<string> listOfImageNames;

  // check for sufficient arguments
  if( argc < 2) {
    printf("usage: %s <directory path>\n", argv[0]);
    exit(-1);
  }

  // get the directory path
  // strcpy(dirname, argv[1]);
  printf("Processing directory %s\n", dirname );

  // open the directory
  dirp = opendir( dirname );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  // loop over all the files in the image file listing
  while( (dp = readdir(dirp)) != NULL ) {

    // check if the file is an image
    if( strstr(dp->d_name, ".jpg") ||
	strstr(dp->d_name, ".png") ||
	strstr(dp->d_name, ".ppm") ||
	strstr(dp->d_name, ".tif") ) {

      printf("processing image file: %s\n", dp->d_name);

      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);
      std::string xyz(buffer);
      listOfImageNames.push_back(xyz);
      // std::cout << "String: " << xyz << std::endl;
      // printf("full path name: %s\n", xyz);

    }
  }
  
  printf("Terminating\n");

  return(0);
}


