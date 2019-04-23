// Fast fourier transform functions (taken from earlier HW assignment)
// Yahia Bakour and James Dunn, Boston University
// EC526 - Parallel Programming final project
// April/May 2019

#include "fft.h"
#include "utilities.h"

// Calculate all the omegas
void makePhase(Complex *omega, int N )
{
    for(int k = 0; k < N; k++)
        omega[k] = exp(2.0*PI*I*(double)k/(double) N);
}

// Slow FT
void FT(Complex * Ftilde, Complex * F, Complex * omega, int N)
{
    for(int k = 0; k < N; k++)
    {
        Ftilde[k] = 0.0;
        for(int x = 0; x < N; x++)
            Ftilde[k] += pow(omega[k],x)*F[x];
    }
}

// Slow inverse FT
void FTinv(Complex * F, Complex * Ftilde, Complex * omega, int N)
{
    for(int x = 0; x < N; x++)
    {
        F[x] = 0.0;
        for(int k = 0; k < N; k++)
            F[x] +=pow(omega[k],-x)*Ftilde[k]/(double) N;
    }
}

// Sample out every other index
// isodd = 0 if evens, 1 if odds
void downsample(Complex *fine, Complex *coarse, int Nfine, int isodd) {
    for (int i=0; i<Nfine/2; ++i) coarse[i] = fine[2*i+isodd];
}



// Recursive FFT version. The mathematical recursive trick that
// we play here is taking advantage of:
// FFT(F) = FFT(F[even_indices]) + e^(2*pi*i*k/N)*FFT(F[odd_indices])
// We recurse down to coarser level "combs" until we reach the individual
// element level, at which point the recursion back up starts.
// Modeled after https://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B
void FFT_(Complex * F, int N) {
    // Need to stop here and return to recurse back up if we are down to a single element.
    // Intuitively, this means that the FFT of a single complex value is equal to itself!
    if (N == 1) return;

    // Split the array in half: evens and odds
    Complex *evens = new Complex [N/2];
    Complex *odds  = new Complex [N/2];
    downsample(F, evens, N, 0);
    downsample(F, odds,  N, 1);

    // Run the FFT on the evens and odds independetly (recurse down)
    FFT_(evens, N/2);
    FFT_(odds,  N/2);
    
    // Sum up the even and odd pieces to form the full FT for this level's elements
    // This actually performs the 
    // FFT(F) = FFT(F[even_indices]) + e^(2*pi*i*k/N)*FFT(F[odd_indices])
    // Don't use the omega calculated earlier, because it has the wrong N in the exp's denominator.
    for(int k = 0; k < N/2; k++) {
        F[k]     = evens[k] + exp(2.0*PI*I*(double)k/(double) N) * odds[k];
        F[k+N/2] = evens[k] - exp(2.0*PI*I*(double)k/(double) N) * odds[k]; // use the F[k+N/2] = odds switched trick
    }


    // Cleanup, no longer need evens and odds arrays
    delete [] evens;
    delete [] odds;
}
// Wraps the FFT_ function and does the fftshift at the end
void FFT(Complex *F, int N) {
    FFT_(F,N);
    //fftshift(F,N);
}

// Recursive inverse FFT version. Same as forward FFT, but opposite sign in the
// exponent and divide by 2 (the recursed application of N/2 divide-by-2's gives the
// desired divide-by-N that is done in the non-recursive FT);
void FFTinv(Complex *F, int N) {
    FFTinv_(F, N);
    for (int i=0;i<N;++i) F[i] /= N;
    fftshift(F,N);
}
void FFTinv_(Complex * F, int N)
{
    // Need to stop here and return to recurse back up if we are down to a single element
    // Intuitively, this means that the FFT of a single complex value is equal to itself!
    if (N == 1) return;

    // Split the array in half: evens and odds
    Complex *evens = new Complex [N/2];
    Complex *odds  = new Complex [N/2];
    downsample(F, evens, N, 0);
    downsample(F, odds,  N, 1);

    // Run the FFT on the evens and odds independetly (recurse down)
    FFTinv_(evens, N/2);
    FFTinv_(odds,  N/2);
    
    // Sum up the even and odd pieces to form the full FT for this level's elements
    // This actually performs the 
    // FFT(F) = FFT(F[even_indices]) + e^(2*pi*i*k/N)*FFT(F[odd_indices])
    // Don't use the omega calculated earlier, because it has the wrong N in the exp's denominator.
    for(int k = 0; k < N/2; k++) {
        F[k]     = (evens[k] + exp(-2.0*PI*I*(double)k/(double) N) * odds[k]);
        F[k+N/2] = (evens[k] - exp(-2.0*PI*I*(double)k/(double) N) * odds[k]); // use the F[k+N/2] = odds switched trick
    }

    // Cleanup, no longer need evens and odds arrays
    delete [] evens;
    delete [] odds;
}



// Print the input array of complex numbers in a nice format
void printComplexArray(FILE *fp, Complex * F, int N) {
    for(int k = 0; k < N; k++) {
        //cout<<"k "<< k << "  F " <<  zeroround(F[k].real()) << " " << zeroround(F[k].imag()) << endl;
        fprintf(fp, "k %2d   F %9.5f %9.5f\n", k, zeroround(F[k].real()), zeroround(F[k].imag()));
    }
}

// Round to zero if less than 1e-10. Makes output prettier
double zeroround(double a) {
    return fabs(a) > 1e-10 ? a : 0.0; 
}

// After an FFT, the elements are in an undesired order. Use this function to shift them to be
// in order of increasing k
void fftshift(Complex * F, int N) {
    // Make a temporary copy of the array
    Complex * tmp = new Complex [N];
    for (int i=0; i<N;++i) tmp[i] = F[i];
    
    // Do the shift
    for (int i=0;i<N;++i) F[i] = tmp[(i+N/2)%N];
    
    delete [] tmp;    
}

// 2D version. Just do a 1D FFT on each row, then on each column.
void FFT2D(Complex **F, int N) {
    int i;
    // Run the FFT across each row
    for (i=0;i<N;++i) FFT(F[i],N);

    // Transpose F to make rows into columns
    transpose(F, N);

    // Run the FFT across each column
    for (i=0;i<N;++i) FFT(F[i],N);

    // Transpose back
    transpose(F, N);
}

// 2D version. Just do a 1D FFT on each row, then on each column.
void FFTinv2D(Complex **F, int N) {
    int i;
    // Run the invFFT across each row
    for (i=0;i<N;++i) FFTinv(F[i],N);

    // Transpose F to make rows into columns
    transpose(F, N);

    // Run the invFFT across each column
    for (i=0;i<N;++i) FFTinv(F[i],N);

    // Transpose back
    transpose(F, N);
}

// Transpose a 2d array of complex numbers
void transpose(Complex **F, int N) {
    // Make a temporary copy of F
    int i, j;
    Complex **tmp = new Complex* [N];
    for (j=0;j<N;++j) tmp[j] = new Complex [N];
    for (i=0;i<N;++i) for (j=0;j<N;++j) tmp[i][j] = F[i][j];

    // Execute the transpose
    for (i=0;i<N;++i) for (j=0;j<N;++j) F[i][j] = tmp[j][i];

    // Cleanup
    for (i=0;i<N;++i) delete [] tmp[i];
    delete [] tmp;
}

// Function to print (real part of) a 2D complex array (as a matrix) for easy input into gnuplot
void printComplexArray2D(FILE* fp, Complex ** F, int N) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            fprintf(fp, "%f ", zeroround(F[i][j].real()));
        }
        fprintf(fp,"\n");
    }
}


// Execute a 2D convolution using the Fourier convolution theorem method:
// conv(A,B) = invFFT(FFT(A) * FFT(B))
// Kernel is assumed to be smaller than image and will be zeropadded to the same size
// image. Arrays are y-major. Convolution is done in-place, the result is put into image.
void FFTImageConvolution(uint8_t *image, int ny, int nx, uint8_t *kernel, int nky, int nkx){
    // Allocate and zeropad kernel
    uint8_t *bigKernel = new uint8_t [nx*ny];
    for (int i=0;i<nx*ny;++i) bigKernel[i] = 0;
    for (int j=0;j<nky;++j) for (int i=0;i<nkx;++i) bigKernel[yxc(j,i,0,nx,1)];

    // Convert to complex 2D raveled

    // FFT kernel and image independently
   
    // Multiply element-by-element

    // Inverse FFT

    // Convert back to unraveled uint8_t

    delete [] bigKernel;
}
