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

//#include "utilities.h"
//#include "utilities.cpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


using std::cout;
using std::endl;

#include <cstdint>

// Hardcoded parameters
#define JPG_QUALITY 100 // 0 to 100, 100 being best quality and largest file
#define NCOLORS 3 // use 3 colors (RGB)

// Preprocessor directives
#define Time std::chrono::time_point<std::chrono::steady_clock>
#define DeltaTime std::chrono::duration<double>

// Forward declarations
void Grayscale(uint8_t *input, uint8_t *output, int ny, int nx, int nc);
void shrink(uint8_t *input, uint8_t *output, int ny, int nx, int nc, int factor);
void enlarge(uint8_t *input, uint8_t *output, int ny, int nx, int nc, int factor);
inline int yxc(int y, int x, int c, int nx, int nc) { return nx*nc*y + nc*x + c; } // converts 3-d indices into 1d index


// Parameters
const uint8_t EDGE_THRESHOLD = 200; // only pixels with gradients larger than this marked as edges

// Forward declarations
void findMultiscaleEdges(uint8_t *input, uint8_t **output, int *levels, int nlevels, int ny, int nx, int nc);
void findEdges(uint8_t *input, uint8_t *output, int ny, int nx, int nc);

// Main execution function
int main(int argc, char ** argv) {
    #pragma acc init
    if (argc != 2) {cout << "Usage: ./edgeDetect [imagefile.jpg]" << endl;return 0;}     // Check for correct usage

    // TO DO LIST:
    // Run with ACC only - be careful about unnecessary memcopy's
    // Run with MPI only (note kernels are arbitrarily sized, so need to be smart about the boundaries!)
    // Run with the FFT and multiply method



    // Print out call sequence that was used
    cout << "Call sequence: ";
    for (int i=0; i<argc; ++i) cout << argv[i] << " ";
    cout << endl;

    // Read in the image we will do edge detection on using stb library
    int nx, ny, nc;
    uint8_t * image = stbi_load(argv[1], &nx, &ny, &nc, NCOLORS); // NCOLORS forces NCOLORS channels per pixel
    cout << "Successfully read " << argv[1] << endl;
    cout << "(nx,ny,nChannels) = (" << nx << "," << ny << "," <<  nc << ")" << endl;


    // Convert image to greyscale for edge detection
    uint8_t * image_gray = new uint8_t [NCOLORS*nx*ny]; // same size as image but only one color channel
    for (long i=0;i<NCOLORS*nx*ny;++i) image_gray[i] = 0;
    cout << "Converting to grayscale...";
    Grayscale(image, image_gray, ny, nx, nc);
    cout << "Done" << endl;

    // Allocate edgemap
    uint8_t * edges = new uint8_t [nx*ny]; // same size as image but only one color channel
    for (long i=0;i<nx*ny;++i) edges[i] = 0;

    // Get the starting timestamp. 
    Time begin_time = std::chrono::steady_clock::now();


    // Run edge detection function
    cout << "Running edge detection...";
    findEdges(image_gray, edges, ny, nx, nc);
    cout << "Done" << endl;


    // Get the end timestamp
    Time end_time = std::chrono::steady_clock::now(); 
    DeltaTime dt = end_time - begin_time; // Compute the difference.
    printf("Edge detection runtime was %.10f seconds\n", dt.count());

    // Write out resulting edgemap
    stbi_write_jpg("edges.jpg", nx, ny, 1, edges, JPG_QUALITY);
    cout << "Wrote edges.jpg" << endl;




    // =================================================================================================== //
    // MULTISCALE EDGE DETECTION
    int nlevels = 2;
    int levels [nlevels];
    levels[0]=1;
    levels[1]=2;

    // Allocate multiscale edgemaps
    uint8_t ** multiscaleEdges = new uint8_t * [nlevels];
    for (int l=0;l<nlevels;++l) multiscaleEdges[l] = new uint8_t [nx*ny/(levels[l]*levels[l])];

    // Get the starting timestamp. 
    Time mbegin_time = std::chrono::steady_clock::now();

    // Run multiscale edge detection
    cout << "Running multiscale edge detection...";
    findMultiscaleEdges(image_gray, multiscaleEdges, levels, nlevels, ny, nx, nc);
    cout << "Done" << endl;


    // Get the end timestamp
    Time mend_time = std::chrono::steady_clock::now(); 
    DeltaTime mdt = mend_time - mbegin_time; // Compute the difference.
    printf("Multiscale Edge detection runtime was %.10f seconds\n", mdt.count());

    // Write out multiscale edgemap images
    uint8_t * enlargedEdges = new uint8_t [ny*nx];
    for (int i=0;i<ny*nx;++i) enlargedEdges[i] = 0;
    for (int l=0;l<nlevels;++l) {
        int factor = levels[l];
        enlarge(multiscaleEdges[l], enlargedEdges, ny/factor, nx/factor, 1, factor);
        char edgeOutfile [20]; 
        sprintf(edgeOutfile,"edges_%dx.jpg", factor);
        stbi_write_jpg(edgeOutfile, nx, ny, 1, enlargedEdges, JPG_QUALITY);
        cout << "Wrote " << edgeOutfile << endl;
    }

    // ==================================================================

    // Cleanup
    stbi_image_free(image);  
    delete [] edges; 
    delete [] image_gray;
    for (int i=0;i<nlevels;++i) delete [] multiscaleEdges[i];
    delete [] multiscaleEdges; 
    delete [] enlargedEdges;
    return 0;
}


// Find edges at various coarser resolution levels. Output must be preallocated.
void findMultiscaleEdges(uint8_t *input, uint8_t **output, int *levels, int nlevels, int ny, int nx, int nc) {

    // Find edges at each of the downsampling levels in levels array and place into output
    for (int l=0;l<nlevels;++l) {
        int factor = levels[l];
        // Shrink image to smaller level
        uint8_t *small_img = new uint8_t [ny*nx*nc/(factor*factor)];
        shrink(input, small_img, ny, nx, nc, factor);

        // Detect edges of the shrunk image
        findEdges(small_img, output[l], ny/factor, nx/factor, nc);
        
        delete [] small_img;
    }

}


// Find the edges in the image at the current resolution using the input kernel (size nkx-by-nky), 
// Output must be preallocated and the same size as input.
void findEdges(uint8_t *pixels, uint8_t *output, int ny, int nx, int nc) {
    int GX [3][3]; int GY [3][3];

    //Sobel Horizontal Mask     
    GX[0][0] = 1; GX[0][1] = 0; GX[0][2] = -1; 
    GX[1][0] = 2; GX[1][1] = 0; GX[1][2] = -2;  
    GX[2][0] = 1; GX[2][1] = 0; GX[2][2] = -1;

    //Sobel Vertical Mask   
    GY[0][0] =  1; GY[0][1] = 2; GY[0][2] =   1;    
    GY[1][0] =  0; GY[1][1] = 0; GY[1][2] =   0;    
    GY[2][0] = -1; GY[2][1] =-2; GY[2][2] =  -1;

    int valX,valY,MAG;
    #pragma acc data copy(output[0:nx*ny*1]) copyin(pixels[0:nx*ny*nc]) copyin(GX[0:3][0:3]) copyin(GY[0:3][0:3]) copyin(valY) copyin(valX) copyin(EDGE_THRESHOLD)
    #pragma acc parallel loop 
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

            // Apply threshold to gradient
            if (MAG > EDGE_THRESHOLD) MAG = 255; else MAG = 0;
            
            //setting the new pixel value
            output[yxc(i,j,0,nx,1)] = MAG;

        }
    }
 
    return;
}


// Converts image into greyscale (maintaining 3 color channels)
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

// Shrink input by an integer factor using a simple average pooling. Output must be allocated already.
// nx and ny are the sizes of the larger input image.
void shrink(uint8_t *input, uint8_t *output, int ny, int nx, int nc, int factor) {
    // Loop over every pixel in the smaller output image, averaging over the nearest "factor" pixels
    // in each direction in the input image.
    // Note: if nx or ny is not evenly divisible by factor, this will leave the rightmost and/or
    // bottommost pixels unaveraged
    int nysml = ny/factor;
    int nxsml = nx/factor;
    uint32_t value = 0;
    for (int ysml=0;ysml<nysml;++ysml) { // loop over columns in output
        for (int xsml=0;xsml<nxsml;++xsml) { // loop over rows in output
            for (int c=0;c<nc;++c) { // loop over color channels
                value = 0;
                for (int yf=0;yf<factor;++yf) { // loop over col pixels within pool
                    for (int xf=0;xf<factor;++xf) { // loop over row pixels within pool
                        value += input[yxc(ysml*factor+yf,xsml*factor+xf,c,nx,nc)];
                    }
                }
                output[yxc(ysml,xsml,c,nxsml,nc)] = value/(factor*factor);
            }
        }
    }

    return;
}

// Enlarge the image by an integer factor by simply copying (maybe do interpolation at some point)
// Output must be allocated already.
void enlarge(uint8_t *input, uint8_t *output, int ny, int nx, int nc, int factor) {
    // Loop over every pixel in the smaller input image and replicate into the larger image
    int nylrg = ny*factor;
    int nxlrg = nx*factor;
    for (int y=0;y<nylrg;++y) { // loop over pixels in the large image
        for (int x=0;x<nxlrg;++x) {
            for (int c=0;c<nc;++c) { // loop over colors
                int ysml = y/factor;
                int xsml = x/factor;
                output[yxc(y,x,c,nxlrg,nc)] = input[yxc(ysml,xsml,c,nx,nc)];
            }
        }
    }

    return;
}
