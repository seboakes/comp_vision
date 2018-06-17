
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

int main()
{
	Mat src = imread("dart4.jpg", 0);

	Mat dst, cdst;
	Canny(src, dst, 50, 200, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);

	vector<Vec4i> lines;
	// detect the lines
	HoughLinesP(dst, lines, 1, (CV_PI / 180), 130, 20, 10);
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		// draw the lines
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);
	}

	imshow("source", src);
	imshow("detected lines", cdst);
	imwrite("Houghlines.jpg", cdst);

	waitKey();

	return 0;
}