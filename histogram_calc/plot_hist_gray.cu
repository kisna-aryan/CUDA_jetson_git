#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
 
using namespace std;
using namespace cv;

#define NoOfBins 65536
 
int main()
{
    // Mat image = imread("nature.jpg", 0);
    Mat image = imread("images/frameIndex_0.png", -1);

    if( image.empty() )
    {
        cout << "Image not Found" << endl;
        return EXIT_FAILURE;
    }
    // allcoate memory for no of pixels for each intensity value
    /*     The maximum number of pixels can be total number of pixels in image.

    Total number of pixels in image resolution 640x512 is = 327680 

    The number of bins in 16 bit image is => 2^16 = 65536
    */
    int histogram[NoOfBins];
 
    // initialize all intensity values to 0
    for(int i = 0; i < NoOfBins; i++)
    {
        histogram[i] = 0;
    }
    
    // cout << sizeof(unsigned short)<< endl;
    cout << "pixel value:" << image.at<u_int16_t>(100,100)<< endl;
    cout <<" Channels:" << image.channels()<< endl;

    // calculate the no of pixels for each intensity values
    for(int y = 0; y < image.rows; y++)
    {
        for(int x = 0; x < image.cols; x++)
        {
            histogram[(int)image.at<u_int16_t>(y,x)]++;
        }
    }
 
    // draw the histograms
    int hist_w = 512; int hist_h = 400;
    // int bin_w = cvRound((double) hist_w/NoOfBins);
    double bin_w = (double) hist_w/NoOfBins;

 
    Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));
 
    // find the maximum intensity element from histogram
    int max = histogram[0];
    for(int i = 1; i < NoOfBins; i++){
        if(max < histogram[i]){
            max = histogram[i];
        }
    }
 
    // normalize the histogram between 0 and histImage.rows
 
    for(int i = 0; i < NoOfBins; i++){
        histogram[i] = ((double)histogram[i]/max)*histImage.rows;
    }
 
 
    // draw the intensity line for histogram
    for(int i = 0; i < NoOfBins; i++)
    {
        line(histImage, Point(cvRound(bin_w*(i)), hist_h),
                              Point(cvRound(bin_w*(i)), hist_h - histogram[i]),
             Scalar(0,0,0), 1, 8, 0);
    }
 
    // display histogram
    namedWindow("Intensity Histogram");
    imshow("Intensity Histogram", histImage);
 
    namedWindow("Image");
    imshow("Image", image);
    waitKey();
    return 0;
}