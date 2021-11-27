#include <cuda.h>

    __global__ void createLUT()
    {
        for(int i = 0; i < NoOfBins; i++)
        {
                new_pixel = (i - min)/min_max_diff;
                if(i >= max)
                {
                    new_pixel = 1;
                }
                else if(i <= min)
                {
                    new_pixel = 0;
                }
                histogram_LUT[i] = (unsigned char)(new_pixel*255);
        }

    } 



__global__ void applyAGC()
{
    // convert image into 8 bit
    // calculate the no of pixels for each intensity values

    for(int y = 0; y < image.rows; y++)
    {
        for(int x = 0; x < image.cols; x++)
        {
            proc_image.at<uchar>(y,x) = histogram_LUT[image.at<u_int16_t>(y,x)];
        }
    }
}
    

 