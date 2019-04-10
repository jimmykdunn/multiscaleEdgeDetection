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
void findEdges(uint8_t *input, bool *kernel, uint8_t *output, int nx, int ny, int nc, int nkx, int nky);
void shrink(uint8_t *input, uint8_t *output, int nx, int ny, int nc, int factor);
void enlarge(uint8_t *input, uint8_t *output, int nx, int ny, int nc, int factor);

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
            for (int chan=0; chan<nc; ++chan) {
                image[nx*nc*y + nc*x + chan] = 255;
            }
        }
    }

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
void findEdges(uint8_t *input, bool *kernel, uint8_t *output, int nx, int ny, int nc, int nkx, int nky) {
    // STUB
    return;
}

// Shrink input by an integer factor using a simple average pooling. Output must be allocated already
void shrink(uint8_t *input, uint8_t *output, int nx, int ny, int nc, int factor) {
    // STUB
    return;
}

// Enlarge the image by an integer factor by simply copying (maybe do interpolation at some point)
// Output must be allocated already.
void enlarge(uint8_t *input, uint8_t *output, int nx, int ny, int nc, int factor) {
    // STUB
    return;
}
