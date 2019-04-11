// Code for multiscale edge detection using openMP (ACC) within MPI ranks
// Yahia Bakour and James Dunn, Boston University
// EC526 - Parallel Programming final project
// April/May 2019
// Image reading/writing code is courtesy of this open source library:
// https://github.com/nothings/stb

#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Hardcoded parameters
#define JPG_QUALITY 100 // 0 to 100, 100 being best quality and largest file
#define NCOLORS 3 // use 3 colors (RGB)


using std::cout;
using std::endl;

// Forward declarations
void findEdges(uint8_t *input, bool *kernel, uint8_t *output, int ny, int nx, int nc, int nky, int nkx);
void shrink(uint8_t *input, uint8_t *output, int ny, int nx, int nc, int factor);
void enlarge(uint8_t *input, uint8_t *output, int ny, int nx, int nc, int factor);
inline int yxc(int y, int x, int c, int nx, int nc) { return nx*nc*y + nc*x + c; }


// Main execution function
int main(int argc, char ** argv) {

    // Check for correct usage
    if (argc != 2) {
        cout << "Usage: ./edgeDetect [imagefile.jpg]" << endl;
        return 0;
    }

    cout << "Call sequence: " << endl;
    for (int i=0; i<argc; ++i) cout << argv[i] << " ";
    cout << endl;

    // Read in the image we will do edge detection on using stb library
    int nx, ny, nc;
    uint8_t * image = stbi_load(argv[1], &nx, &ny, &nc, NCOLORS); // NCOLORS forces NCOLORS channels per pixel
    cout << "(nx,ny,nChannels) = (" << nx << "," << ny << "," <<  nc << ")" << endl;


    // A FEW EXAMPLES SHOWING HOW TO ACCESS AND CHANGE INDIVIDUAL PIXEL VALUES
    // Make the top row of pixels pure yellow
    for (int x=0; x<nx; ++x) {
        image[nc*x + 0] = 255;
        image[nc*x + 1] = 255;
    }

    // Notice that we are y-major here, meaning that pixel 1 is (0,0) but pixel 2 is (1,0)
    // Accordingly, it is much faster to access a row of pixels at once than a column of pixels
    // Make a few rows of pixels pure white
    for (int y=100; y<110; ++y){
        for (int x=0; x<nx; ++x) {
            for (int c=0; c<nc; ++c) {
                //image[nx*nc*y + nc*x + c] = 255;
                image[yxc(y,x,c,nx,nc)] = 255;
            }
        }
    }

    // Test shrink function
    int factor = 4;
    uint8_t * imagesmall = new uint8_t [ny*nx*nc];
    shrink(image,imagesmall,ny,nx,nc,4);
    stbi_write_jpg("outputSmall4x.png", nx/factor, ny/factor, nc, imagesmall, JPG_QUALITY);
    delete [] imagesmall;


    // Things to do:
    // Fill out all STUB functions
    // Import and use timing functions to compare all of the following methods for various image sizes
    // Run with ACC only - be careful about unnecessary memcopy's
    // Run with MPI only (note kernels are arbitrarily sized, so need to be smart about the boundaries!)
    // Run with the FFT and multiply method


    // Write it back out to a file, in a few formats
    stbi_write_png("output.png", nx, ny, nc, image, nx*3);
    stbi_write_jpg("output.jpg", nx, ny, nc, image, JPG_QUALITY);
    stbi_write_bmp("output.bmp", nx, ny, nc, image);


    // Cleanup
    stbi_image_free(image);    
    return 0;
}

// Find the edges in the image at the current resolution using the input kernel (size nkx-by-nky)
// Output must be preallocated and the same size as input.
void findEdges(uint8_t *input, bool *kernel, uint8_t *output, int ny, int nx, int nc, int nky, int nkx) {
    // STUB
    return;
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
    // STUB
    return;
}
