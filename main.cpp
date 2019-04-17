// Code for multiscale edge detection using openMP (ACC) within MPI ranks
// Yahia Bakour and James Dunn, Boston University
// EC526 - Parallel Programming final project
// April/May 2019
// Image reading/writing code is courtesy of this open source library:
// https://github.com/nothings/stb

#include <iostream>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "utilities.h"


using std::cout;
using std::endl;

// Forward declarations
void findEdges(uint8_t *input, uint8_t *output, int ny, int nx, int nc);
void Grayscale(uint8_t *input, uint8_t *output, int ny, int nx, int nc);

// Main execution function
int main(int argc, char ** argv) {

    // Check for correct usage
    if (argc != 2) {
        cout << "Usage: ./edgeDetect [imagefile.jpg]" << endl;
        return 0;
    }

    // Print out call sequence that was used
    cout << "Call sequence: " << endl;
    for (int i=0; i<argc; ++i) cout << argv[i] << " ";
    cout << endl;

    // Read in the image we will do edge detection on using stb library
    int nx, ny, nc;
    uint8_t * image = stbi_load(argv[1], &nx, &ny, &nc, NCOLORS); // NCOLORS forces NCOLORS channels per pixel
    cout << "(nx,ny,nChannels) = (" << nx << "," << ny << "," <<  nc << ")" << endl;

    // Actually do the edge detection now

    // Things to do:
    // Run with ACC only - be careful about unnecessary memcopy's
    // Run with MPI only (note kernels are arbitrarily sized, so need to be smart about the boundaries!)
    // Run with the FFT and multiply method

    // Convert image to greyscale for edge detection
    uint8_t * image_gray = new uint8_t [NCOLORS*nx*ny]; // same size as image but only one color channel
    for (long i=0;i<NCOLORS*nx*ny;++i) image_gray[i] = 0;
    Grayscale(image, image_gray, ny, nx, nc);

    // Allocate edgemap
    uint8_t * edges = new uint8_t [nx*ny]; // same size as image but only one color channel
    for (long i=0;i<nx*ny;++i) edges[i] = 0;

    // Get the starting timestamp. 
    std::chrono::time_point<std::chrono::steady_clock> begin_time =
        std::chrono::steady_clock::now();


    // Run edge detection function (currently a stub that does nothing)
    findEdges(image_gray, edges, ny, nx, nc);


    // Get the end timestamp
    std::chrono::time_point<std::chrono::steady_clock> end_time =
        std::chrono::steady_clock::now(); // Get the ending timestamp.
    std::chrono::duration<double> difference_in_time = end_time - begin_time; // Compute the difference.
    double difference_in_seconds = difference_in_time.count(); // Get the difference in seconds.
    printf("Edge detection runtime was %.10f seconds\n", difference_in_seconds);

    // Write out resulting edgemap
    stbi_write_jpg("edges.jpg", nx, ny, 1, edges, JPG_QUALITY);
    cout << "Wrote edges.jpg" << endl;

    // Cleanup
    stbi_image_free(image);  
    delete [] edges;  
    return 0;
}

void Grayscale(uint8_t *pixels, uint8_t *output, int ny, int nx, int nc) {
    //Calculating the grayscale in each pixel. 
    int val1,val2,val3;
    //The values of the 3 colours (R, B and G) are all the same  
    for(int i=0; i < ny; i++)
        {
            for(int j=0; j < nx; j++)
            {
                val1 = pixels[yxc(i,j,0,nx,nc)];
                val2=val1;
                val3=val1;
                output[yxc(i,j,0,nx,nc)] = val1;
                output[yxc(i,j,1,nx,nc)] = val2;
                output[yxc(i,j,2,nx,nc)] = val3;
            }
        }

}


// Find the edges in the image at the current resolution using the input kernel (size nkx-by-nky), Output must be preallocated and the same size as input.
void findEdges(uint8_t *pixels, uint8_t *output, int ny, int nx, int nc) {
    unsigned int GX [3][3]; unsigned int GY [3][3];

    //Sobel Horizontal Mask     
    GX[0][0] = 1; GX[0][1] = 0; GX[0][2] = -1; 
    GX[1][0] = 2; GX[1][1] = 0; GX[1][2] = -2;  
    GX[2][0] = 1; GX[2][1] = 0; GX[2][2] = -1;

    //Sobel Vertical Mask   
    GY[0][0] =  1; GY[0][1] = 2; GY[0][2] =   1;    
    GY[1][0] =  0; GY[1][1] = 0; GY[1][2] =   0;    
    GY[2][0] = -1; GY[2][1] =-2; GY[2][2] =  -1;

    int valX,valY,MAG;
    for(int i=0; i < ny; i++)
    {
        valX = 0;valY = 0;
        for(int j=0; j < nx; j++)
        {
            //setting the pixels around the border to 0, because the Sobel kernel cannot be allied to them
            if ((i==0)||(i==ny-1)||(j==0)||(j==nx-1))
            {valX=0;valY=0;}
            else
            {
                valX = 0;
                valY = 0;
                for (int x = -1; x <= 1; x++){
                    for (int y = -1; y <= 1; y++)
                    {
                        //image[nx*nc*y + nc*x + c] = 255;
                        valX = valX +  pixels[yxc(i+x,j+y,0,nx,nc)]* GX[1+x][1+y];
                        valY = valY +  pixels[yxc(i+x,j+y,0,nx,nc)]* GY[1+x][1+y];
                    }
                }
            }
            //Gradient magnitude
            MAG = sqrt(valX*valX + valY*valY);
            //setting the new pixel value
            output[yxc(i,j,0,nx,1)] = MAG;
        }
    }
 
    return;
}
