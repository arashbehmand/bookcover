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
Point UL,UR,LL,LR;
int taskLatch=-1;

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

void extractPoints(int,void*)
{
	Mat hsv,srcbl,gscl,dst,cdst;
	src=src(Rect(2,2,src.cols-2,src.rows-2));
	int curdim=max(src.rows,src.cols);
	int ndim=700;
	if(curdim>ndim)
	{
		Size sz;
		if(src.rows>src.cols)
		{
			sz=Size((src.cols*ndim)/src.rows,ndim);
		}else
		{
			sz=Size(ndim,(src.rows*ndim)/src.cols);
		}
		resize(src,src,sz);
	}
	GaussianBlur(src,srcbl,cv::Size(15,15),1.9,1.9);

	vector<Mat> hsv_plane;
	//
	//srcbl=src.clone();
	Mat canny_output;

	Canny(srcbl, canny_output, 2,50,3);
	if(DEBUG & 0x01) imshow("c1",canny_output);
	
	rectangle(canny_output,Rect(0,0,canny_output.cols,canny_output.rows),Scalar(0),7);
	if(DEBUG & 0x01) imshow("canny",canny_output);
	if(DEBUG & 0x02) imwrite("dbg/01-canny1.jpg",canny_output);
	cvtColor(canny_output, cdst, CV_GRAY2BGR);


  vector<Vec4i> lines;
  HoughLinesP(canny_output, lines, 4, 30*CV_PI/180, 50, 25, 40 );
  for( size_t i = 0; i < lines.size(); i++ )
  {
    Vec4i l = lines[i];
    line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
  }

  if(DEBUG & 0x01) imshow("detected lines",cdst);
  if(DEBUG & 0x02) imwrite("dbg/02-ht1.jpg",cdst);
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
		if(DEBUG & 0x03) drawContours( drawing1, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		double area=contourArea(hull[i]);
		if(area>Max)
		{
			Max=area;
			MaxI=i;
		}
	}	
	if(DEBUG & 0x01) imshow("cv hu",drawing1);
	if(DEBUG & 0x02) imwrite("dbg/03-cvxhul.jpg",drawing1);
	Mat maskOvly = Mat::zeros( src.size(), CV_8UC1 );

	drawContours(maskOvly,hull,MaxI,Scalar(255),40,8);
	if(DEBUG & 0x02) imwrite("dbg/04-mask.jpg",maskOvly);
	Mat newCny;
	bitwise_and(canny_output,maskOvly,newCny);
	if(DEBUG & 0x01) imshow("newCny",newCny);
	if(DEBUG & 0x02) imwrite("dbg/05-canny2.jpg",newCny);

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
	if(DEBUG & 0x01) imshow("lines",newCny);
	if(DEBUG & 0x02) imwrite("dbg/06-ht2.jpg",newCny);

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
	UL=Point(-1,-1);
	LL=Point(-1,-1);
	UR=Point(-1,-1);
	LR=Point(-1,-1);
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
	if(DEBUG & 0x01) imshow("!",fnl);
	if(DEBUG & 0x02) imwrite("dbg/07-marks.jpg",fnl);
	cntr=0.25*cntr;
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

	/*
	/// Detector parameters
	int blockSize = 10;
	int apertureSize = 5;
	double k = 0.05;
	Mat cd,cdn,cdns;
	cvtColor(srcbl,cd,CV_BGR2GRAY);
	Mat maskOvlyCD1 = Mat::zeros( src.size(), CV_8UC1 );
	Mat maskOvlyCD2 = Mat::zeros( src.size(), CV_8UC1 );
	circle(maskOvlyCD1, UL, 20, Scalar(255), 41);
	circle(maskOvlyCD1, UR, 20, Scalar(255), 41);
	circle(maskOvlyCD1, LL, 20, Scalar(255), 41);
	circle(maskOvlyCD1, LR, 20, Scalar(255), 41);
	circle(maskOvlyCD2, UL, 10, Scalar(255), 21);
	circle(maskOvlyCD2, UR, 10, Scalar(255), 21);
	circle(maskOvlyCD2, LL, 10, Scalar(255), 21);
	circle(maskOvlyCD2, LR, 10, Scalar(255), 21);
	GaussianBlur(maskOvlyCD1,maskOvlyCD1,Size(31,31),15,15);
	imshow("cd_g",cd);
#undef min
	//min(cd,maskOvlyCD1,cd);
	imshow("cd",cd);
	/// Detecting corners
	cornerHarris( cd, cd, blockSize, apertureSize, k, BORDER_DEFAULT );

	/// Normalizing
	normalize( cd, cdn, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
	convertScaleAbs( cdn, cdns );
	cdns.convertTo(cdns,CV_8UC1);
	imshow("cdns",cdns);
	min(cdns,maskOvlyCD2,cdn);
	cdns=cdn.clone();
	/// Drawing a circle around corners
	for( int j = 0; j < cdn.rows ; j++ )
	{
		for( int i = 0; i < cdn.cols; i++ )
		{
			if( (int) cdn.at<uchar>(j,i) >	30 )
			{
				circle( cdns, Point( i, j ), 5,  Scalar(255), 2, 8, 0 );
			}
		}
	}*/
	//imshow("cdns",cdns);
  Mat cd;
  int blockSize = 3; int apertureSize = 3;
  int myShiTomasi_qualityLevel = 1;
  int max_qualityLevel = 100;
  double myShiTomasi_minVal; double myShiTomasi_maxVal;
  cvtColor(srcbl,cd,CV_BGR2GRAY);
  Mat myShiTomasi_dst = Mat::zeros( cd.size(), CV_32FC1 );
  cornerMinEigenVal( cd, myShiTomasi_dst, blockSize, apertureSize, BORDER_DEFAULT );
  minMaxLoc( myShiTomasi_dst, &myShiTomasi_minVal, &myShiTomasi_maxVal, 0, 0, Mat() );  
  normalize(myShiTomasi_dst,myShiTomasi_dst,0,255,NORM_MINMAX,CV_8UC1);

  /*if( myShiTomasi_qualityLevel < 1 ) { myShiTomasi_qualityLevel = 1; }
  
  for( int j = 0; j < myShiTomasi_dst.rows; j++ )
     { for( int i = 0; i < myShiTomasi_dst.cols; i++ )
          {
            if( myShiTomasi_dst.at<float>(j,i) > myShiTomasi_minVal + ( myShiTomasi_maxVal - myShiTomasi_minVal )*myShiTomasi_qualityLevel/max_qualityLevel )
              { myShiTomasi_copy.at<uchar>(j,i)=(uchar)((myShiTomasi_dst.at<float>(j,i)-myShiTomasi_minVal)/( myShiTomasi_maxVal - myShiTomasi_minVal )*255); }
          }
     }*/
  Mat myShiTomasi_chs[3],myShiTomasi_RGB;
	myShiTomasi_chs[0]=myShiTomasi_dst.clone();
	myShiTomasi_chs[1]=myShiTomasi_dst.clone();
	myShiTomasi_chs[2]=myShiTomasi_dst.clone();
	
#undef min
#undef max
	Mat maskOvlyCD;
	double maxv;
	int maxIdx[2];
	maskOvlyCD = Mat::zeros( src.size(), CV_8UC1 );
	circle(maskOvlyCD, UL, 5, Scalar(255), 11);
	GaussianBlur(maskOvlyCD,maskOvlyCD,Size(11,11),10,10);
	if(DEBUG) max(myShiTomasi_chs[1],maskOvlyCD,myShiTomasi_chs[1]);
	min(myShiTomasi_dst,maskOvlyCD,maskOvlyCD);
	minMaxIdx(maskOvlyCD,0,&maxv,0,maxIdx);
	if(norm(UL-Point(maxIdx[1],maxIdx[0]))<10)
	{
		if(DEBUG) circle(myShiTomasi_chs[2],Point(maxIdx[1],maxIdx[0]),3,Scalar(255),2);
		cout<<"UL"<<endl;
		UL=Point(maxIdx[1],maxIdx[0]);
	}
	
	maskOvlyCD = Mat::zeros( src.size(), CV_8UC1 );
	circle(maskOvlyCD, UR, 5, Scalar(255), 11);
	GaussianBlur(maskOvlyCD,maskOvlyCD,Size(11,11),10,10);
	if(DEBUG) max(myShiTomasi_chs[1],maskOvlyCD,myShiTomasi_chs[1]);
	min(myShiTomasi_dst,maskOvlyCD,maskOvlyCD);
	minMaxIdx(maskOvlyCD,0,&maxv,0,maxIdx);
	if(norm(UR-Point(maxIdx[1],maxIdx[0]))<10)
	{
		if(DEBUG) circle(myShiTomasi_chs[2],Point(maxIdx[1],maxIdx[0]),3,Scalar(255),2);
		cout<<"UR"<<endl;
		UR=Point(maxIdx[1],maxIdx[0]);
	}

	maskOvlyCD = Mat::zeros( src.size(), CV_8UC1 );
	circle(maskOvlyCD, LL, 5, Scalar(255), 11);
	GaussianBlur(maskOvlyCD,maskOvlyCD,Size(11,11),10,10);
	if(DEBUG) max(myShiTomasi_chs[1],maskOvlyCD,myShiTomasi_chs[1]);
	min(myShiTomasi_dst,maskOvlyCD,maskOvlyCD);
	minMaxIdx(maskOvlyCD,0,&maxv,0,maxIdx);
	if(norm(LL-Point(maxIdx[1],maxIdx[0]))<10)
	{
		if(DEBUG) circle(myShiTomasi_chs[2],Point(maxIdx[1],maxIdx[0]),3,Scalar(255),2);
		cout<<"LL"<<endl;
		LL=Point(maxIdx[1],maxIdx[0]);
	}

	maskOvlyCD = Mat::zeros( src.size(), CV_8UC1 );
	circle(maskOvlyCD, LR, 5, Scalar(255), 11);
	GaussianBlur(maskOvlyCD,maskOvlyCD,Size(11,11),10,10);
	if(DEBUG) max(myShiTomasi_chs[1],maskOvlyCD,myShiTomasi_chs[1]);
	min(myShiTomasi_dst,maskOvlyCD,maskOvlyCD);
	minMaxIdx(maskOvlyCD,0,&maxv,0,maxIdx);
	if(norm(LR-Point(maxIdx[1],maxIdx[0]))<10)
	{
		if(DEBUG) circle(myShiTomasi_chs[2],Point(maxIdx[1],maxIdx[0]),3,Scalar(255),2);
		cout<<"LR"<<endl;
		LR=Point(maxIdx[1],maxIdx[0]);
	}

	merge(myShiTomasi_chs,3,myShiTomasi_RGB);
	
	if(DEBUG & 0x01) imshow("shithomasi",myShiTomasi_RGB);
	if(DEBUG & 0x02) imwrite("dbg/08-shithomasi.jpg",myShiTomasi_RGB);
}

void interactPoints(int,void*)
{
	Mat fnl=src.clone();
	circle(fnl, UL, 3, Scalar(0  , 255,   0), 3);
	circle(fnl, UR, 3, Scalar(255, 255,   0), 3);
	circle(fnl, LL, 3, Scalar(0  , 255, 255), 3);
	circle(fnl, LR, 3, Scalar(255,   0, 255), 3);
	imshow("Fine Tune",fnl);
}

void extractBook(int,void*)
{
	extrxtd=Mat(600,420,CV_8UC3);
	cout<<extrxtd.rows<<" "<<extrxtd.cols<<endl;
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
	//if(DEBUG & 0x01) 
	imshow("extrxtd",extrxtd);
}

void mouseInteractPoints( int event, int x, int y, int flags, void* param ){
	Point p(x,y);
	switch( event ){
	case CV_EVENT_LBUTTONDOWN:
		if(norm((p-UL))<10)
			taskLatch=1;
		else if(norm((p-UR))<10)
			taskLatch=2;
		else if(norm((p-LL))<10)
			taskLatch=3;
		else if(norm((p-LR))<10)
			taskLatch=4;
		else
			taskLatch=-1;
		printf("(%d,%d) TL=%d\r\n",x,y,taskLatch);
		break;
	case CV_EVENT_LBUTTONUP:
		switch(taskLatch)
		{
			case 1: UL=p; break;
			case 2: UR=p; break;
			case 3: LL=p; break;
			case 4: LR=p; break;
		}
		taskLatch=-1;
		interactPoints(0,0);
		printf("(%d,%d)\r\n",x,y);
		break;
	case CV_EVENT_RBUTTONDOWN:
		extractBook(0,0);
		break;
	}
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
	

	if(DEBUG & 0x01) cvNamedWindow("Source Image");
	if(DEBUG & 0x01) cvSetMouseCallback("Source Image", mousePositionOut, &src);
	if(DEBUG & 0x01) imshow("Source Image",src);

	extractPoints(0,0);
	cvNamedWindow("Fine Tune");
	cvSetMouseCallback("Fine Tune", mouseInteractPoints, &src);
	interactPoints(0,0);

	//if(DEBUG & 0x01) 
	waitKey(0);
	extractBook(0,0);
	imwrite("out.jpg",extrxtd);
	return 0;
}