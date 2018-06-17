//header inclusion
#include <stdio.h> 
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream> 
#include <math.h>


using namespace cv;
using namespace std;

void Sobel(
  cv::Mat & input,
  int size,
  cv::Mat & blurredOutput,
  cv::Mat & blurredOutputDy, cv::Mat & gradient, cv::Mat & direction);

String cascade_name = "cascade.xml";    //classifier to use
CascadeClassifier cascade;

int main(int argc, char * * argv) {  

  // LOADING THE IMAGE
  char * imageName = argv[1];
  Mat image;
  image = imread(imageName, 1);

  if (argc != 2 || !image.data) {
    printf(" No image data \n ");
    return -1;
  }
  if (!cascade.load(cascade_name)) {
    printf("--(!)Error loading\n");
    return -1;
  };

  Mat gray_image;
  cvtColor(image, gray_image, CV_BGR2GRAY);   //convert the colored image to gray

  Mat coinDy;					//Matrix variables to use for sobel function below and normalization
  Mat coinDx;
  Mat gradientFinal;
  Mat normalizedDx;
  Mat normalizedDy;
  Mat directionResult;
  Mat directionFinal;
  Mat gradientResult;
  Mat frame = image.clone();     //used for viola jones

  Sobel(gray_image, 3, coinDx, coinDy, gradientResult, directionResult);   //calling the sobel operative function with parameters

  cv::normalize(coinDy, normalizedDy, 0, 255, NORM_MINMAX, CV_8UC1);				//normalizing
  cv::normalize(gradientResult, gradientFinal, 0, 255, NORM_MINMAX, CV_8UC1);		//normalizing
  cv::normalize(directionResult, directionFinal, 0, 255, NORM_MINMAX, CV_8UC1);	    //normalizing


  ////////////////////////////////// Viola Jones Detector//////////////////////////////////////////////

  std::vector < Rect > darts; 
  Mat frame_gray;
  cvtColor(frame, frame_gray, CV_BGR2GRAY);
  equalizeHist(frame_gray, frame_gray);
  cascade.detectMultiScale(frame_gray, darts, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));
  
  //////////////////////////////////hough circle/////////////////////////////////////////////////////

  int min_radius = 0;
  int max_radius = 20;
  int vote = 0;
  Mat HoughOutput;

  int ThreeD_Array[] = {
    gradientResult.rows,
    gradientResult.cols,
    (max_radius - min_radius)
  };							//creating a 3D Array 

  Mat ThreeD_space = Mat(3, ThreeD_Array, CV_32S, Scalar(0));						//creating a 3D matrix which is empty and has the size of ThreeD_Array
  Mat HoughSpace(gradientResult.rows, gradientResult.cols, CV_32F, Scalar(0));		//creating a 2D matrix which is empty and is the size of gradientResult image
  Mat threshold(gradientResult.rows, gradientResult.cols, CV_8UC1, Scalar(0));			//creating a 2D matrix which is empty and is the size of gradientResult image

  // Threshold by looping through all pixels

  for (int y = 0; y < gradientResult.rows; y++) {
    for (int x = 0; x < gradientResult.cols; x++) {
      if (gradientResult.at < float > (y, x) > 200) {
        threshold.at < uchar > (y, x) = 255;
      } else {
        threshold.at < uchar > (y, x) = 0;
      }
    }
  }
  
  imwrite("threshold.jpg",threshold);  //store an image named "threshold.jpg" which is threshold in the program.

  ////////////////////////////// performing the Hough Space Circle Transform/////////////////// 

////////////////////////////// equation 1 : - -  /////////////////////

  for (int x = 0; x < threshold.rows; x++) {
    for (int y = 0; y < threshold.cols; y++) {
      if (threshold.at < uchar > (x, y) > 0) {
        for (int r = 0; r < (max_radius - min_radius); r++) {

          int a = (int)(x - (r + min_radius) * cos(directionResult.at < float > (x, y)));
          int b = (int)(y - (r + min_radius) * sin(directionResult.at < float > (x, y)));

          if ((a >= 0 && a < threshold.rows) && (b >= 0 && b < threshold.cols)) {
            ThreeD_space.at < int > (a, b, r) += 1;

          }
        }
      }
    }
  }
//////////////////////////////////equation 2: - + ///////////////////////////

  for (int x = 0; x < threshold.rows; x++) {
    for (int y = 0; y < threshold.cols; y++) {
      if (threshold.at < uchar > (x, y) > 0) {
        for (int r = 0; r < (max_radius - min_radius); r++) {

          int a = (int)(x - (r + min_radius) * cos(directionResult.at < float > (x, y)));
          int b = (int)(y + (r + min_radius) * sin(directionResult.at < float > (x, y)));
          if ((a >= 0 && a < threshold.rows) && (b >= 0 && b < threshold.cols)) {
            ThreeD_space.at < int > (a, b, r) += 1;

          }
        }
      }
    }
  }

 //////////////////////////////////equation 3: + - ///////////////////////////

  for (int x = 0; x < threshold.rows; x++) {
    for (int y = 0; y < threshold.cols; y++) {
      if (threshold.at < uchar > (x, y) > 0) {
        for (int r = 0; r < (max_radius - min_radius); r++) {

          int a = (int)(x + (r + min_radius) * cos(directionResult.at < float > (x, y)));
          int b = (int)(y - (r + min_radius) * sin(directionResult.at < float > (x, y)));
          if ((a >= 0 && a < threshold.rows) && (b >= 0 && b < threshold.cols)) {
            ThreeD_space.at < int > (a, b, r) += 1;
          }
        }
      }
    }
  }

//////////////////////////////////equation 4: + + ///////////////////////////

  for (int x = 0; x < threshold.rows; x++) {
    for (int y = 0; y < threshold.cols; y++) {
      if (threshold.at < uchar > (x, y) > 0) {
        for (int r = 0; r < (max_radius - min_radius); r++) {
          int a = (int)(x + (r + min_radius) * cos(directionResult.at < float > (x, y)));
          int b = (int)(y + (r + min_radius) * sin(directionResult.at < float > (x, y)));
          if ((a >= 0 && a < threshold.rows) && (b >= 0 && b < threshold.cols)) {
            ThreeD_space.at < int > (a, b, r) += 1;

          }
        }
      }
    }
  }

  vector < int > houghDetection_x;			//create vectors to store the pixel values
  vector < int > houghDetection_y;
  int MaxVote=0;							//declaring a variable to set the vote threshold

  for (int x = 0; x < threshold.rows; x++) {	
    for (int y = 0; y < threshold.cols; y++) {
      for (int r = 0; r < (max_radius - min_radius); r++) {
        if (ThreeD_space.at < int > (x, y, r) > vote) {    //incrementing the vote variable when the condition is match
          vote += 1;
        }
        if (ThreeD_space.at < int > (x, y, r) >MaxVote) {  //varibale used to set the vote threshold
			MaxVote=ThreeD_space.at < int > (x, y, r);
        }
      }
    }
  }

  for (int x = 0; x < threshold.rows; x++) {	
    for (int y = 0; y < threshold.cols; y++) {
      for (int r = 0; r < (max_radius - min_radius); r++) {
        if (ThreeD_space.at < int > (x, y, r) >=(MaxVote-5)) {  
          houghDetection_x.push_back(y);   //store the pixel values in the vectors
          houghDetection_y.push_back(x);
        }
      }
    }
  }

  for (int x = 0; x < threshold.rows; x++) {
    for (int y = 0; y < threshold.cols; y++) {
      for (int r = 0; r < (max_radius - min_radius); r++) {
        HoughSpace.at < float > (x, y) += ThreeD_space.at < int > (x, y, r);   //creating a hough space
      }
    }

  }

  for (int i = 0; i < darts.size(); i++)  //draw the frames around the detected dartboards -- Viola Jones
  {
    rectangle(frame, Point(darts[i].x, darts[i].y), Point(darts[i].x + darts[i].width, darts[i].y + darts[i].height), Scalar(0, 255, 0), 2);
  }

  for (int i = 0; i < darts.size(); i++) {     //distance between the pixels from the vector and the centre of the frame produced by viola jones
    for (int j = 0; j < houghDetection_x.size(); j++) {
      int s = pow((houghDetection_x[j] - (darts[i].x + (darts[i].width / 2))), 2) + pow((houghDetection_y[j] - (darts[i].y + (darts[i].height / 2))), 2);  
      int distance = abs(sqrt(s));

      if (distance <= 20) {
        rectangle(image, Point(darts[i].x, darts[i].y), Point(darts[i].x + darts[i].width, darts[i].y + darts[i].height), Scalar(255, 255, 0), 2);   //draw a frame around the matched pixel value
      }
    }
  }

  normalize(HoughSpace, HoughOutput, 0, 255, CV_MINMAX);  //normalizing
  imwrite("HoughSpace.jpg", HoughOutput);   //storing the images 
  imwrite("IntegartedImage.jpg", image);
  imwrite("ViolaJones.jpg", frame);

  return 0;
}

void Sobel(cv::Mat & input, int size, cv::Mat & blurredOutput, cv::Mat & blurredOutputDy, cv::Mat & gradient, cv::Mat & direction) {

  blurredOutput.create(input.size(), CV_32F);   //creating matrices with float data type
  blurredOutputDy.create(input.size(), CV_32F);
  gradient.create(input.size(), CV_32F);
  direction.create(input.size(), CV_32F);

  // create the Gaussian kernel in 1D 
  cv::Mat kX = cv::getGaussianKernel(size, -1);
  cv::Mat kY = cv::getGaussianKernel(size, -1);

  // make it 2D multiply one by the transpose of the other
  cv::Mat kernel = kX * kY.t();
  cv::Mat kernelDy = kX * kY.t();

  //CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
  //TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

  // we need to create a padded version of the input
  // or there will be border effects
  int kernelRadiusX = (kernel.size[0] - 1) / 2;
  int kernelRadiusY = (kernel.size[1] - 1) / 2;

  kernel.at < double > (0, 0) = (double) - 1;   //dx kernel
  kernel.at < double > (0, 1) = (double) 0;
  kernel.at < double > (0, 2) = (double) 1;
  kernel.at < double > (1, 0) = (double) - 2;
  kernel.at < double > (1, 1) = (double) 0;
  kernel.at < double > (1, 2) = (double) 2;
  kernel.at < double > (2, 0) = (double) - 1;
  kernel.at < double > (2, 1) = (double) 0;
  kernel.at < double > (2, 2) = (double) 1;

  kernelDy.at < double > (0, 0) = (double) - 1;     //dy kernel
  kernelDy.at < double > (0, 1) = (double) - 2;
  kernelDy.at < double > (0, 2) = (double) - 1;
  kernelDy.at < double > (1, 0) = (double) 0;
  kernelDy.at < double > (1, 1) = (double) 0;
  kernelDy.at < double > (1, 2) = (double) 0;
  kernelDy.at < double > (2, 0) = (double) 1;
  kernelDy.at < double > (2, 1) = (double) 2;
  kernelDy.at < double > (2, 2) = (double) 1;

  cv::Mat paddedInput;
  cv::copyMakeBorder(input, paddedInput,     //create a border 
    kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
    cv::BORDER_REPLICATE);

  // now we can do the convoltion
  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      double sum = 0.0;

      for (int m = -kernelRadiusX; m <= kernelRadiusX; m++) {
        for (int n = -kernelRadiusY; n <= kernelRadiusY; n++) {
          // find the correct indices we are using
          int imagex = i + m + kernelRadiusX;
          int imagey = j + n + kernelRadiusY;
          int kernelx = m + kernelRadiusX;
          int kernely = n + kernelRadiusY;

          // get the values from the padded image and the kernel
          int imageval = (int) paddedInput.at < uchar > (imagex, imagey);
          double kernalval = kernel.at < double > (kernelx, kernely);

          // do the multiplication
          sum += imageval * kernalval;
        }
      }

      blurredOutput.at < float > (i, j) = (float) sum;
    }
  }

  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      double sum = 0.0;
      for (int m = -kernelRadiusX; m <= kernelRadiusX; m++) {
        for (int n = -kernelRadiusY; n <= kernelRadiusY; n++) {
          // find the correct indices we are using
          int imagex = i + m + kernelRadiusX;
          int imagey = j + n + kernelRadiusY;
          int kernelx = m + kernelRadiusX;
          int kernely = n + kernelRadiusY;

          // get the values from the padded image and the kernel
          int imageval = (int) paddedInput.at < uchar > (imagex, imagey);
          double kernalval = kernelDy.at < double > (kernelx, kernely);

          // do the multiplication
          sum += imageval * kernalval;
        }
      }

      // set the output value as the sum of the convolution
      blurredOutputDy.at < float > (i, j) = (float) sum;
    }
  }



  //////gradient 
  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      float a = pow(blurredOutput.at < float > (i, j), 2);
      float b = pow(blurredOutputDy.at < float > (i, j), 2);
      float c = a + b;
      float d = sqrt(c);
      gradient.at < float > (i, j) = (float) d;
    }
  }

  //direction
  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      float e = atan2(blurredOutputDy.at < float > (i, j), blurredOutput.at < float > (i, j));
      direction.at < float > (i, j) = (float) e;
    }
  }
}