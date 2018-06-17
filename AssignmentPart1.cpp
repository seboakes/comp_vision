//ooooh look at me i can use Github - Seb, 2018



//inlcude libraries
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>

using namespace std;
using namespace cv;

/** Function Headers */
int image1();
int image2();
int image3();
int image4();
int image5();


/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;


/** @function main */
int main()
{
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	image1(); //calling the function
	image2(); //calling the function
	image3(); //calling the function
	image4(); //calling the function
	image5(); //calling the function

	return 0;
}



int image1()
{
	Mat frame = imread("dart4.jpg", CV_LOAD_IMAGE_COLOR);    //performing the viola jones detection
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
	int facesDetected=faces.size();   //obtaining the no of faces detected

	//Ground Truth Details
	int groundTruthFaces=1;   //defining the ground truth
	int x=319;
	int y=57;
	int height=221;
	int width=176;
	//
	int groundCentreX=x+(width/2);    //centre point x for the ground truth
	int groundCentreY=y+(height/2);   //centre point y for the ground truth
	int faceMatch=0;					//variable to hold the no of matches

	for (int i=0;i<faces.size();i++)    //loops through the faces detected
	{
		int detectedCentreX=faces[i].x+(faces[i].width/2);           //centre point x for the detected face
		int detectedCentreY=faces[i].y+(faces[i].height/2);			 //centre point y for the detected face

		for (int j=0;j<groundTruthFaces;j++)     //loop to check if there is a match
		{
			int s =pow((groundCentreX-detectedCentreX),2) +pow((groundCentreY-detectedCentreY),2);    //using the formula to calculate the distance from detected centre point point to ground truth centre point
			int distance=abs(sqrt(s));
	
			if (distance<=50)       //threshold for the distance between them to be considered a match
			{
				faceMatch=faceMatch+1;
			}
		}
	}
	cout<<"\n"<<endl;
	cout<<"Detected no of faces for Dart 4: "<< facesDetected<<endl;  
	cout<<"Dart 4 has "<<faceMatch << " match/es"<<endl;
	float TP=(float)faceMatch;								//defining the True Positive
	float FP=(float)facesDetected-(float)faceMatch;         //defining the False Positive
	float FN=(float)groundTruthFaces-(float)TP;             //defining the False Negative
	float TPR=TP/(TP+FN);	                                //defining the True Positive Rate
	float F1=(2*TP)/((2*TP)+FP+FN);							//definign the F1 score
	cout<<"TPR: "<<TPR<<endl;
	cout<<"F1: "<<F1<<endl;
	cout<<"\n"<<endl;

	return 0;				
}

int image2()
{	
	Mat frame_2 = imread("dart5.jpg", CV_LOAD_IMAGE_COLOR);		//performing the viola jones detection
	std::vector<Rect> faces_2;
	Mat frame_gray_2;
	cvtColor( frame_2, frame_gray_2, CV_BGR2GRAY );
	equalizeHist( frame_gray_2, frame_gray_2 );
	cascade.detectMultiScale( frame_gray_2, faces_2, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
	int facesDetected_2=faces_2.size();

	int groundTruthFaces_2=11;
	int x_2[11]={61,58,184,246,287,369,419,508,550,636,675};
	int y_2[11]={138,234,200,163,239,180,229,178,239,189,245};
	int height_2[11]={64,59,71,66,69,77,71,71,78,73,64};
	int width_2[11]={72,90,86,70,73,72,78,69,79,71,68};

	int groundCentreX_2[11]={0};
	int groundCentreY_2[11]={0};

	for (int i=0;i<11;i++)
	{
		groundCentreX_2[i]=x_2[i]+(width_2[i]/2);
		groundCentreY_2[i]=y_2[i]+(height_2[i]/2);
	}
	 int faceMatch_2=0;


	 for (int i=0;i<facesDetected_2;i++)
	{
		int detectedCentreX_2=faces_2[i].x+(faces_2[i].width/2);
		int detectedCentreY_2=faces_2[i].y+(faces_2[i].height/2);

		for (int j=0;j<groundTruthFaces_2;j++)
		{
		
			int s_2 =pow((groundCentreX_2[j]-detectedCentreX_2),2) + pow((groundCentreY_2[j]-detectedCentreY_2),2);
			int distance_2=abs(sqrt(s_2));
		
			if (distance_2<=50)
				{
					faceMatch_2=faceMatch_2+1;
				}
		}
	}
	cout<<"Detected no of faces for Dart 5: "<< facesDetected_2<<endl;
	cout<<"Dart 5 has "<<faceMatch_2 << " match/es"<<endl;
	float TP=(float)faceMatch_2;
	float FP=(float)facesDetected_2-(float)faceMatch_2;
	float FN=(float)groundTruthFaces_2-(float)faceMatch_2;
	float TPR=TP/(TP+FN);	
	float F1=(2*TP)/((2*TP)+FP+FN);
	cout<<"TPR: "<<TPR<<endl;
	cout<<"F1: "<<F1<<endl;
	cout<<"\n"<<endl;
	return 0;
}

int image3()
{

	Mat frame_3 = imread("dart13.jpg", CV_LOAD_IMAGE_COLOR);			//performing the viola jones detection
	std::vector<Rect> faces_3; 
	Mat frame_gray_3;
	cvtColor( frame_3, frame_gray_3, CV_BGR2GRAY );
	equalizeHist( frame_gray_3, frame_gray_3 );
	cascade.detectMultiScale( frame_gray_3, faces_3, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
	int facesDetected_3=faces_3.size();

	int groundTruthFaces_3=1;
	int x_3=410;
	int y_3=133;
	int height_3=130;
	int width_3=127;
	//
	int groundCentreX_3=x_3+(width_3/2);
	int groundCentreY_3=y_3+(height_3/2);

	int faceMatch_3=0;

	for (int i=0;i<faces_3.size();i++)
	{
		int detectedCentreX_3=faces_3[i].x+(faces_3[i].width/2);
		int detectedCentreY_3=faces_3[i].y+(faces_3[i].height/2);

		for (int j=0;j<groundTruthFaces_3;j++)
		{
			int s_3 =pow((groundCentreX_3-detectedCentreX_3),2) + pow((groundCentreY_3-detectedCentreY_3),2);
			int distance_3=abs(sqrt(s_3));

			if (distance_3<=50)
			{
				faceMatch_3=faceMatch_3+1;
			}
		}
	}
	cout<<"Detected no of faces for Dart 13: "<< facesDetected_3<<endl;
	cout<<"Dart 13 has "<<faceMatch_3 << " match/es"<<endl;
	float TP=(float)faceMatch_3;
	float FP=(float)facesDetected_3-(float)faceMatch_3;
	float FN=(float)groundTruthFaces_3-(float)faceMatch_3;
	float TPR=TP/(TP+FN);	
	float F1=(2*TP)/((2*TP)+FP+FN);
	cout<<"TPR: "<<TPR<<endl;
	cout<<"F1: "<<F1<<endl;
	cout<<"\n"<<endl;
	return 0;

}

int image4()
{
	Mat frame_4 = imread("dart14.jpg", CV_LOAD_IMAGE_COLOR);			//performing the viola jones detection
	std::vector<Rect> faces_4;
	Mat frame_gray_4;
	cvtColor( frame_4, frame_gray_4, CV_BGR2GRAY );
	equalizeHist( frame_gray_4, frame_gray_4 );
	cascade.detectMultiScale( frame_gray_4, faces_4, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
	int facesDetected_4=faces_4.size();

	int groundTruthFaces_4=2;
	int x_4[2]={461,721};
	int y_4[2]={227,196};
	int height_4[2]={99,107};
	int width_4[2]={97,114};

	int groundCentreX_4[2]={0};
	int groundCentreY_4[2]={0};

	for (int i=0;i<2;i++)
	{
		groundCentreX_4[i]=x_4[i]+(width_4[i]/2);
		groundCentreY_4[i]=y_4[i]+(height_4[i]/2);

	}
	 int faceMatch_4=0;

	 for (int i=0;i<facesDetected_4;i++)
	{
		int detectedCentreX_4=faces_4[i].x+(faces_4[i].width/2);
		int detectedCentreY_4=faces_4[i].y+(faces_4[i].height/2);

		for (int j=0;j<groundTruthFaces_4;j++)
		{
			int s_4 =pow((groundCentreX_4[j]-detectedCentreX_4),2) + pow((groundCentreY_4[j]-detectedCentreY_4),2);
			int distance_4=abs(sqrt(s_4));

			if (distance_4<=50)
				{
					faceMatch_4=faceMatch_4+1;
				}
		}
	}
	cout<<"Detected no of faces for Dart 14: "<< facesDetected_4<<endl;
	cout<<"Dart 14 has "<<faceMatch_4 << " match/es"<<endl;
	float TP=(float)faceMatch_4;
	float FP=(float)facesDetected_4-(float)faceMatch_4;
	float FN=(float)groundTruthFaces_4-(float)faceMatch_4;
	float TPR=TP/(TP+FN);	
	float F1=(2*TP)/((2*TP)+FP+FN);
	cout<<"TPR: "<<TPR<<endl;
	cout<<"F1: "<<F1<<endl;
	cout<<"\n"<<endl;
	return 0;

}

int image5()
{
	Mat frame_5 = imread("dart15.jpg", CV_LOAD_IMAGE_COLOR);				//performing the viola jones detection
	std::vector<Rect> faces_5;
	Mat frame_gray_5;
	cvtColor( frame_5, frame_gray_5, CV_BGR2GRAY );
	equalizeHist( frame_gray_5, frame_gray_5 );
	cascade.detectMultiScale( frame_gray_5, faces_5, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
	int facesDetected_5=faces_5.size();
	//Ground Truth Details
	int groundTruthFaces_5=0;
	int faceMatch_5=0;
	cout<<"Detected no of faces for Dart 15: "<< facesDetected_5<<endl;
	cout<<"Dart 15 has "<<faceMatch_5 << " match/es"<<endl;
	float TP=(float)faceMatch_5;
	float FP=(float)facesDetected_5-(float)faceMatch_5;
	float FN=(float)groundTruthFaces_5-(float)faceMatch_5;
	float TPR=TP/(TP+FN);	
	float F1=(2*TP)/((2*TP)+FP+FN);
	cout<<"TPR: "<<TPR<<endl;
	cout<<"F1: "<<F1<<endl;
	cout<<"\n"<<endl;
	return 0;
}
