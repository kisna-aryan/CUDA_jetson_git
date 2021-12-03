#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
 
using namespace std;
using namespace cv;

#define NoOfBins 65536
#define maxThresold 10
#define minThersold 10

int histogram[NoOfBins];
unsigned char histogram_LUT[NoOfBins];
unsigned char *d_histogram_ptr;
unsigned int *d_hist_max, *d_hist_min;

__global__ void creatLUT(unsigned char *d_histogram, unsigned int *hist_min, unsigned int *hist_max)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    float min_max_diff = *hist_max - *hist_min;
    int d_NoOfBins = 65536; 
    if (idx < d_NoOfBins)
    {
        float new_pixel = (idx - *hist_min)/min_max_diff;
        if(idx >= *hist_max)
        {
            new_pixel = 1;
        }
        else if(idx <= *hist_min)
        {
            new_pixel = 0;
        }
        d_histogram[idx] = (unsigned char)(new_pixel*255);
    }
}

__global__ void applyAGC(unsigned short *src_img, unsigned char  *proc_image, unsigned char *d_histogram, int img_rows, int img_cols)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

     proc_image[y*img_cols + x] = d_histogram[src_img[y*img_cols + x]];
}
 
int main()
{
    // Mat image = imread("nature.jpg", 0);
    Mat image = imread("images/frameIndex_0.png", -1);

    if( image.empty() )
    {
        cout << "Image not Found" << endl;
        return EXIT_FAILURE;
    }
    Mat proc_image = Mat::zeros(Size(image.cols,image.rows),CV_8UC1);

    // Create two temporary images (for holding sobel gradients)
    unsigned char *d_process_img;
    unsigned short *d_original_image;
    cudaMalloc(&d_original_image, image.cols * image.rows* sizeof(unsigned short));
    cudaMalloc(&d_process_img, image.cols* image.rows* sizeof(unsigned char));
    cudaMalloc(&d_histogram_ptr, NoOfBins);

    cudaMemcpy(d_original_image, image.data, image.rows * image.cols* sizeof(unsigned short), cudaMemcpyHostToDevice);

    // allcoate memory for no of pixels for each intensity value
    /*     The maximum number of pixels can be total number of pixels in image.

    Total number of pixels in image resolution 640x512 is = 327680 

    The number of bins in 16 bit image is => 2^16 = 65536
    */

 
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
    int hist_max = histogram[0];
    for(int i = 1; i < NoOfBins; i++){
        if(hist_max < histogram[i]){
            hist_max = histogram[i];
        }
    }
    // find the maximum intensity element from histogram
    int max = histogram[NoOfBins-1];
    for(int i = NoOfBins-2; i > 1; i--){
        if(maxThresold < histogram[i]){
            max = i;
            break;
        }
    }

    // find the minimum intensity element from histogram
    int min = histogram[1];
    for(int i = 2; i < NoOfBins; i++){
        if(minThersold < histogram[i]){
            min = i;
            break;
        }
    }
    // // Fill the LUT with new pixel value
    // float new_pixel, min_max_diff;
    // min_max_diff = max-min;

    // for(int i = 0; i < NoOfBins; i++)
    // {
    //         new_pixel = (i - min)/min_max_diff;
    //         if(i >= max)
    //         {
    //             new_pixel = 1;
    //         }
    //         else if(i <= min)
    //         {
    //             new_pixel = 0;
    //         }
    //         histogram_LUT[i] = (unsigned char)(new_pixel*255);
    // }
    
    // cudaMemcpy(d_histogram_ptr, histogram_LUT, NoOfBins, cudaMemcpyHostToDevice);


    cout << "max:" << max << endl << "min:" << min <<endl;

                // convolution kernel launch parameters
    dim3 cblocks (image.cols / 16, image.rows/ 16);
    dim3 cthreads(16, 16);

    // pythagoran kernel launch paramters
    dim3 pblocks (image.cols * image.rows / 256);
    dim3 pthreads(256, 1);

    cudaMemset(d_histogram_ptr, 0, NoOfBins);
    cudaMemcpy(d_hist_max, &max, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist_min, &min, sizeof(unsigned int), cudaMemcpyHostToDevice);


    creatLUT<<<NoOfBins/256,256>>>(d_histogram_ptr, d_hist_min, d_hist_max);
    cudaThreadSynchronize();

    applyAGC<<<pblocks,pthreads>>>(d_original_image, d_process_img, d_histogram_ptr, 480, 640);
    cudaThreadSynchronize();
    cudaMemcpy(proc_image.data,d_process_img, image.rows * image.cols* sizeof(unsigned char), cudaMemcpyDeviceToHost);


    // normalize the histogram between 0 and histImage.rows
 
    for(int i = 0; i < NoOfBins; i++){
        histogram[i] = ((double)histogram[i]/hist_max)*histImage.rows;
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
    imshow("Image", proc_image);
    waitKey();
    return 0;
}