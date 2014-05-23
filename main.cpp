#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <windows.h>

#define DEBUG 2

using namespace cv;
using namespace std;

Mat src,extrxtd;

void getImageFromDialog(Mat& img)
{
	OPENFILENAME ofn;       // common dialog box structure
	char szFile[260];       // buffer for file name

	// Initialize OPENFILENAME
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.lpstrFile = szFile;
	// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
	// use the contents of szFile to initialize itself.
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = "All\0*.*\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = NULL;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

	// Display the Open dialog box. 

	if (GetOpenFileName(&ofn)==TRUE) 
	{
		img = imread( ofn.lpstrFile);
	}
}

void mousePositionOut( int event, int x, int y, int flags, void* param ){

	switch( event ){
	case CV_EVENT_LBUTTONDOWN:
		
		printf("(%d,%d)\r\n",x,y);
		break;
	}
}

bool acceptLinePair(Vec2f line1, Vec2f line2, float minTheta)
{
    float theta1 = line1[1], theta2 = line2[1];

    if(theta1 < minTheta)
    {
        theta1 += CV_PI; // dealing with 0 and 180 ambiguities...
    }

    if(theta2 < minTheta)
    {
        theta2 += CV_PI; // dealing with 0 and 180 ambiguities...
    }

    return abs(theta1 - theta2) > minTheta;
}

vector<Point2f> lineToPointPair(Vec2f line)
{
    vector<Point2f> points;

    float r = line[0], t = line[1];
    double cos_t = cos(t), sin_t = sin(t);
    double x0 = r*cos_t, y0 = r*sin_t;
    double alpha = 1000;

    points.push_back(Point2f(x0 + alpha*(-sin_t), y0 + alpha*cos_t));
    points.push_back(Point2f(x0 - alpha*(-sin_t), y0 - alpha*cos_t));

    return points;
}

// the long nasty wikipedia line-intersection equation...bleh...
Point2f computeIntersect(Vec2f line1, Vec2f line2)
{
    vector<Point2f> p1 = lineToPointPair(line1);
    vector<Point2f> p2 = lineToPointPair(line2);

    float denom = (p1[0].x - p1[1].x)*(p2[0].y - p2[1].y) - (p1[0].y - p1[1].y)*(p2[0].x - p2[1].x);
    Point2f intersect(((p1[0].x*p1[1].y - p1[0].y*p1[1].x)*(p2[0].x - p2[1].x) -
                       (p1[0].x - p1[1].x)*(p2[0].x*p2[1].y - p2[0].y*p2[1].x)) / denom,
                      ((p1[0].x*p1[1].y - p1[0].y*p1[1].x)*(p2[0].y - p2[1].y) -
                       (p1[0].y - p1[1].y)*(p2[0].x*p2[1].y - p2[0].y*p2[1].x)) / denom);

    return intersect;
}

void extractBook(int,void*)
{
	Mat hsv,srcbl,gscl,dst,cdst;
	vector<Mat> hsv_plane;
	GaussianBlur(src,srcbl,cv::Size(15,15),2,2);
	//cvtColor(srcbl,hsv,CV_BGR2HSV);
	/*
	Mat samples(src.rows * src.cols, 3, CV_32F);
	for( int y = 0; y < src.rows; y++ )
		for( int x = 0; x < src.cols; x++ )
			for( int z = 0; z < 3; z++)
				samples.at<float>(y + x*src.rows, z) = hsv.at<Vec3b>(y,x)[z];


	int clusterCount = 10;
	Mat labels;
	int attempts = 3;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 1000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );

	Mat new_image( src.size(), src.type() );
	for( int y = 0; y < src.rows; y++ )
		for( int x = 0; x < src.cols; x++ )
		{ 
			int cluster_idx = labels.at<int>(y + x*src.rows,0);
			new_image.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
			new_image.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
			new_image.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
		}
	cvtColor(new_image,new_image,CV_HSV2BGR);
	if(DEBUG) imshow( "clustered image", new_image );*/
	//cvtColor(srcbl,gscl,CV_BGR2GRAY);
	//split(hsv,hsv_plane);
	//gscl=new_image.clone();//hsv_plane[0].clone();
	//threshold(gscl,gscl,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
	//if(DEBUG) imshow("gscl",gscl);
	//Mat mu,sigma;
	Mat canny_output;
	//meanStdDev(gscl,mu,sigma);
	Canny(srcbl, canny_output, 5,60,3);
	imshow("c1",canny_output);
	//Mat element = getStructuringElement( 0, Size( 3,3 ), Point( 1,1 ) );
	/// Apply the specified morphology operation
	//morphologyEx( canny_output, canny_output, CV_MOP_OPEN, element );
	//imshow("c2",canny_output);
	rectangle(canny_output,Rect(0,0,canny_output.cols,canny_output.rows),Scalar(0),8);
	if(DEBUG) imshow("canny",canny_output);
	if(DEBUG>1) imwrite("dbg/canny1.jpg",canny_output);
	cvtColor(canny_output, cdst, CV_GRAY2BGR);
 #if 0
  vector<Vec2f> lines;
  HoughLines(dst, lines, 5, 5*CV_PI/180, 150, 0, 0 );

  for( size_t i = 0; i < lines.size(); i++ )
  {
     float rho = lines[i][0], theta = lines[i][1];
     Point pt1, pt2;
     double a = cos(theta), b = sin(theta);
     double x0 = a*rho, y0 = b*rho;
     pt1.x = cvRound(x0 + 1000*(-b));
     pt1.y = cvRound(y0 + 1000*(a));
     pt2.x = cvRound(x0 - 1000*(-b));
     pt2.y = cvRound(y0 - 1000*(a));
     line( cdst, pt1, pt2, Scalar(0,0,255), 1, CV_AA);
  }
 #else
  vector<Vec4i> lines;
  HoughLinesP(canny_output, lines, 5, 30*CV_PI/180, 50, 25, 25 );
  for( size_t i = 0; i < lines.size(); i++ )
  {
    Vec4i l = lines[i];
    line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
  }
 #endif
  if(DEBUG) imshow("detected lines",cdst);
  if(DEBUG>1) imwrite("dbg/ht1.jpg",cdst);
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// Detect edges using canny
  RNG rng(12345);
  /// Find contours
  Mat tmp;
  cvtColor(cdst,tmp,CV_BGR2GRAY);
  findContours(tmp , contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	/// Find the convex hull object for each contour
	vector<vector<Point> >hull( contours.size() );
	for( int i = 0; i < contours.size(); i++ )
		{  convexHull( Mat(contours[i]), hull[i], false ); }

	/// Draw contours + hull results
	Mat drawing1 = Mat::zeros( src.size(), CV_8UC3 );
	int MaxI=-1;
	double Max=0;
	for( int i = 0; i< contours.size(); i++ )
	{
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		//drawContours( drawing1, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		drawContours( drawing1, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		double area=contourArea(hull[i]);
		if(area>Max)
		{
			Max=area;
			MaxI=i;
		}
	}	
	if(DEBUG) imshow("cv hu",drawing1);
	if(DEBUG>1) imwrite("dbg/cvxhul.jpg",drawing1);
	Mat maskOvly = Mat::zeros( src.size(), CV_8UC1 );
	drawContours(maskOvly,hull,MaxI,Scalar(255),40,8);
	if(DEBUG>1) imwrite("dbg/mask.jpg",maskOvly);
	Mat newCny;
	bitwise_and(canny_output,maskOvly,newCny);
	if(DEBUG) imshow("newCny",newCny);
	if(DEBUG>1) imwrite("dbg/canny2.jpg",newCny);
	//vector<Vec4i> linesF;
	//HoughLinesP(newCny, linesF, 1, CV_PI/180, 50, 150, 50 );
	//newCny=Mat::zeros( src.size(), CV_8UC1 );
	//for( size_t i = 0; i < linesF.size(); i++ )
	//{
	//	Vec4i l = linesF[i];
	//	line( newCny, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(128), 1, CV_AA);
	//}
	//imshow("lines",newCny);
	vector<Vec2f> linesF;
	HoughLines(newCny, linesF, 1, 1*CV_PI/180, 50, 0, 0 );
	newCny=Mat::zeros( src.size(), CV_8UC1 );
	for( size_t i = 0; i < linesF.size(); i++ )
	{
		float rho = linesF[i][0], theta = linesF[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 10000*(-b));
		pt1.y = cvRound(y0 + 10000*(a));
		pt2.x = cvRound(x0 - 10000*(-b));
		pt2.y = cvRound(y0 - 10000*(a));
		line( newCny, pt1, pt2, Scalar(255), 1, CV_AA);
	}
	if(DEBUG) imshow("lines",newCny);
	if(DEBUG>1) imwrite("dbg/ht2.jpg",newCny);

	vector<Point2f> intersections;
    for( size_t i = 0; i < linesF.size(); i++ )
    {
        for(size_t j = 0; j < linesF.size(); j++)
        {
            Vec2f line1 = linesF[i];
            Vec2f line2 = linesF[j];
            if(acceptLinePair(line1, line2, CV_PI / 6))
            {
                Point2f intersection = computeIntersect(line1, line2);
				if(intersection.inside(Rect(Point(0,0),src.size())))
					intersections.push_back(intersection);
            }
        }
	}	

	Mat fnl=src.clone();
    if(intersections.size() > 0)
    {
        vector<Point2f>::iterator i;
        for(i = intersections.begin(); i != intersections.end(); ++i)
        {
            //cout << "Intersection is " << i->x << ", " << i->y << endl;
            circle(fnl, *i, 1, Scalar(0, 255, 0), 3);
        }
    }

	Mat interKM(intersections.size(),2,CV_32F),interLB,interC;
	for(int i=0;i<intersections.size();i++)
	{
		interKM.at<float>(i,0)=intersections[i].x;
		interKM.at<float>(i,1)=intersections[i].y;
	}
	if(intersections.size()<4)
	{
		cerr<<"Error: less than 4 intersection points";
		return;
	}
	kmeans(interKM, 4, interLB, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 1000, 0.0001), 5, KMEANS_PP_CENTERS, interC);
	Point cntr;
	for(int i=0;i<4;i++)
	{
		cntr+=Point(interC.at<float>(i,0),interC.at<float>(i,1));
		circle(fnl, Point(interC.at<float>(i,0),interC.at<float>(i,1)), 3, Scalar(0, 0, 255), 3);
	}
	if(DEBUG) imshow("!",fnl);
	if(DEBUG>1) imwrite("dbg/marks.jpg",fnl);
	cntr=0.25*cntr;
	Point UL,UR,LL,LR;
	for(int i=0;i<4;i++)
	{
		if(interC.at<float>(i,1)<cntr.y)
		{
			if(interC.at<float>(i,0)<cntr.x)
			{
				UL=Point(interC.at<float>(i,0),interC.at<float>(i,1));
			}else
			{
				UR=Point(interC.at<float>(i,0),interC.at<float>(i,1));
			}
		}else
		{
			if(interC.at<float>(i,0)<cntr.x)
			{	
				LL=Point(interC.at<float>(i,0),interC.at<float>(i,1));
			}else
			{
				LR=Point(interC.at<float>(i,0),interC.at<float>(i,1));
			}
		}
	}
	extrxtd=Mat(600,420,CV_8UC3);
	cout<<extrxtd.rows<<" "<<extrxtd.cols<<endl;
	cout<<UL<<endl;
	cout<<UR<<endl;
	cout<<LL<<endl;
	cout<<LR<<endl;
	vector<Point2f> cr,ds;
	cr.push_back(Point2f(UL));
	ds.push_back(Point2f(0,0));
	cr.push_back(Point2f(UR));
	ds.push_back(Point2f(extrxtd.cols,0));
	cr.push_back(Point2f(LR));
	ds.push_back(Point2f(extrxtd.cols,extrxtd.rows));
	cr.push_back(Point2f(LL));
	ds.push_back(Point2f(0,extrxtd.rows));

	Mat trnsfrm=getPerspectiveTransform(cr,ds);
	warpPerspective(src,extrxtd,trnsfrm,extrxtd.size());
	if(DEBUG) imshow("extrxtd",extrxtd);
	/*

	{
		vector<Mat> channels;
		split(extrxtd,channels);
        Mat B,G,R;

        equalizeHist( channels[0], B );
        equalizeHist( channels[1], G );
        equalizeHist( channels[2], R );
        vector<Mat> combined;
        combined.push_back(B);
        combined.push_back(G);
        combined.push_back(R);
        Mat result;
        merge(combined,result);
        imshow("eqhist",result);
	}

	{
		vector<Mat> channels;
		split(extrxtd,channels);
        Mat B,G,R;

		Ptr<CLAHE> claheB,claheG,claheR;
		claheB = createCLAHE();
		claheB->setClipLimit(4);
        claheB->apply( channels[0], B );

        claheG = createCLAHE();
		claheG->setClipLimit(4);
        claheG->apply( channels[1], G );

        claheR = createCLAHE();
		claheR->setClipLimit(4);
        claheR->apply( channels[2], R );
        vector<Mat> combined;
        combined.push_back(B);
        combined.push_back(G);
        combined.push_back(R);
        Mat result;
        merge(combined,result);
        imshow("clahe",result);
	}
	*/
  /*/// Detector parameters
  int blockSize = 2;
  int apertureSize = 3;
  double k = 0.04;
  Mat cd,cdn,cdns;
  /// Detecting corners
  cornerHarris( srcbl, cd, blockSize, apertureSize, k, BORDER_DEFAULT );

  /// Normalizing
  normalize( cd, cdn, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  convertScaleAbs( cdn, cdns );

  /// Drawing a circle around corners
  for( int j = 0; j < cdn.rows ; j++ )
     { for( int i = 0; i < cdn.cols; i++ )
          {
            if( (int) cdn.at<float>(j,i) >	100 )
              {
               circle( cdns, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
              }
          }
     }
  imshow("cdns",cdns);*/
}

int main( int argc, char** argv )
{
	srand( (unsigned)time( NULL ) );
	if(argc>1){
		src = imread( argv[1] );
	}else{
		getImageFromDialog(src);
	}
	if( !src.data )
	{
		cout<<"error";
		waitKey(0);
		return -1;
	}
	if(max(src.rows,src.cols)>900)
	{
		Size sz;
		if(src.rows>src.cols)
		{
			sz=Size((src.cols*900)/src.rows,900);
		}else
		{
			sz=Size(900,(src.rows*900)/src.cols);
		}
		resize(src,src,sz);
	}
	src=src(Rect(2,2,src.cols-2,src.rows-2));
	if(DEBUG) cvNamedWindow("Source Image");
	if(DEBUG) cvSetMouseCallback("Source Image", mousePositionOut, &src);
	if(DEBUG) imshow("Source Image",src);

	extractBook(0,0);
	imwrite("out.jpg",extrxtd);

	//if(DEBUG) 
		waitKey(0);
	return 0;
}