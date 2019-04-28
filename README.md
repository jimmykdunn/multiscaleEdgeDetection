# Parallizeing Multiscale Edge Detection with OpenACC, OpenMPI, and FFT

By: Yahia Bakour and James Dunn

## Abstract
Multiscale edge detection is a computer vision technique that finds pixels in an image that have sharp gradients at differing physical scales.  Multiscale edge-maps are useful for tasks such as object segmentation and image alignment and registration.  At the core of all edge detection algorithms is a convolution of the input image with a kernel approximating the spatial derivative (gradient) of the image brightness.  This convolution, as well as other loops in the program running the edge detection algorithm, are prime candidates for a parallel implementation across multiple CPUs or GPUs.  

We present a parallel implementation of the multiscale edge detection algorithm in c++ using openACC.  A separate MPI parallel implementation is also presented. The parallel implementations are compared with both a serial implementation and an implementation that uses the Fourier convolution theorem.
Code was tested on the Boston University shared computing cluster (scc1.bu.edu).

## Introduction

Edge Detection includes a variety of mathematical methods that aim at identifying points in an image where the image brightness changes sharply (has discontinuities). Edge detection is a fundamental tool in computer vision and image processing. We aim to utilize sobel edge detection algorithm [1] to generate an edge map when given an image. The sobel operator uses two 3x3 kernels which are convolved with the original image to calculate approximations of the derivatives for both the horizontal and vertical changes.

The multiscale edge detection procedure simply runs the usual edge detection algorithm at multiple scales.  Specifically, it takes the original image, shrinks it by some factor(s), then runs the same edge detection algorithm on the shrunk image(s).  The result is a stack of edge-maps that show the edges in the image at different physical scales.  The images in Figures 1 through 8 show one potential application of the multiscale technique.

The non-dependent loops in the multiscale edge detection process make it a prime candidate for a parallel implementation, both on a GPU via openACC, and across cores via MPI.

We first implement a serial version of this edge detection algorithm and use it for multiscale edge detection. We then examine two different parallelization techniques with the aim of speeding up the program: an OpenACC implementation run on a GPU, and an MPI implementation with each scale of edges run on a separate core.  Finally, we compare a serial implementation that uses the Fourier convolution theorem to execute the convolutions in the edge detection algorithm.

In all implementations, the output edgemaps themselves are identical or nearly identical, but the runtime is markedly different.  We use runtime as our primary performance metric.

### Prerequisites
> * GCC
> * openMPI
> * openACC
> * We used a shared computing cluser with Tesla GPUs, but you will require a gpu to run this.

### Serial Implementation
The serial implementation of our multiscale edge detection algorithm proceeds as follows.  We first shrink the input image by a factor F[2,16], we run the Sobel Edge Detection algorithm on the shrunk image, then we enlarge the image by a factor F, then repeat for larger and larger factors F. The purpose of this is to gain more and more detail from the image on previously overlooked edges. 

Algorithm for Multiscale Edge Detection:
Require: I : Input Image, Output: Array containing output images, F: Array of factors to run multiscale edge detection on.

```
Multiscale(I,Output,F):
-For I in range(Len(F)):
--OutSmallImage <- Size of output image with factor
--ShrinkImage(Input, OutSmallImage, F[I]) 
--SobelEdgeDetection(OutSmallImage,OutEdgemap)
--EnlargeImage(OutEdgemap, Output[I], F[I])
```
<b>Equation 1: Pseudocode for the multiscale edge detection algorithm. </b>

Sobel Edge Detection is done by convolving the following 2 3x3 kernels with the original image :
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/kernelsforsobeloperator.png" >

<p>Where <b>A</b> is the original input image and Gx and Gy are the gradients found from the convolution. We then find the Magnitude of the gradient from it’s components and decide whether it is above or below a certain threshold. If we determine that it is above the edge threshold, then we count it as an edge. Shrinking the input image is done using simple average pooling to determine what the new pixel should be from the previous pixels. Enlarging the image is done by simple copying of the previous pixel to the new pixels.</p>

### OpenACC Implementation
<p>Very quickly we realized that the serial version of our code had many backwards compatibility issues and we had to rewrite the code for our grayscale, enlarge, shrink, and sobel edge detection functions to be parallelizable by openACC. This led to a speedup of slightly below 5x. Then we realized that a lot of the time, the arrays necessary for GPU accelerated multiscale edge detection were already on the GPU and didn’t have to be copied in and out repetitively. We rewrote our multiscale edge detection function to only copyin the input array (Image) once and then constantly refer to it when needed using openACC’s present clause. We managed to gain a speedup of x7 on the small image from the serial version after modifying it to remove all backwards dependencies and ensuring to copyin all static arrays during the multiscale edge detection. We also turned our 2 kernels from 3x3 matrices to 18 single int variables which helped shave off a few more seconds as we didn’t have to perform a copyin of the 2 kernels everytime the function is run.</p>

<p>One of the main issues we faced was decoupling dependent loops to gain independence from one another, after the loops were intrinsically parallel we would insert the <i><b>#pragma acc loop independent</b></i> directive to tell the compiler that the loops are independent from one another. We also tested using the <i><b>#pragma acc loop collapse(n)</b></i>  directive that collapses the first n loops into one loop but found better results with the first approach.</p>

<p>We also realized that a simple naive ACC Implementation of this project would have only gained a speedup of x2 but after we repeatedly analyzed the code for speed bottlenecks, we were able to maximize our usage of the GPU through the use of OpenACC best practices and accelerate our code. </p>


## OpenMPI Implementation
<p>Open MPI provides a smooth and compact way to simultaneously execute a program on multiple “ranks”.  Each rank can be a separate core, CPU, GPU, or even a separate physical computer.  Open MPI also provides a standard for communication between the ranks that is made invisible to the programmer. </p>

<p>The most obvious and clear-cut way to speed up multiscale edge detection with MPI is to simply run each of the edge scales on its own rank because the scales are completely independent from one another.  A less clear-cut way to use MPI would be to split up the image into blocks and assign each block to a different rank.  Less obvious still would be to use MPI to split up the edge scales across GPUs, and then let openACC parallelize on each GPU.</p>

<p>We implement the first of these methods - simply running each scale in the multiscale edge detection algorithm on its own core of a CPU.  Using MPI and ACC simultaneously to run across multiple GPUs was explored but abandoned for lack of time.</p>

## FFT Implementation

<p>The Fast Fourier Transform’s (FFT) primary use is to quickly convert data from time space to frequency space and back again with order Nlog2(N) operations.  By comparison, the standard (slow) Discrete Fourier Transform (DFT) uses order N2 operations.  The FFT thus provides the potential for significant decreases in runtime, particularly for large arrays.  Unfortunately, the FFT is a recursive algorithm, so it is not amenable to parallelization.</p>

<p>One of the other uses of the FFT is to perform fast convolutions, which could make the convolution step of our edge detection algorithm faster.  This is made possible by use of the Fourier Convolution Theorem [3].  The Fourier Convolution Theorem states that the inverse Fourier transform of the product of two Fourier transformed functions is mathematically equivalent to convolution.  In discrete pseudocode: <b> A⊗B=invFFT(FFT(A) * FFT(B)) </b> </p>
<p>The most intuitive way to understand this is that the shift operation in frequency space is performed with multiplication.  Mathematically, a shift of x by  is performed with:
<b>e^(ik(x+lamba))=e^(ikx)*e(ik*lambda)</b></p>

<p>Convolution is merely a series of shifts and multiplications, so this simple equation brings light to why the Fourier convolution theorem works.</p>

<p>Naïvely, one would expect that simply taking the serial edge detection code and replacing the corresponding 4x nested for loops with a FFT-based convolution would offer an N4 to 2N2log2(N) speedup.  Upon careful consideration however, this arithmetic does not accurately describe the convolutions that are performed in edge detection.  </p>

<p> Let the image that we wish to convolve by be NxN pixels, and let the edge detection kernel be the usual 3x3 Sobel kernel.  One direction of the Sobel convolution (x or y), thus takes only O(9N2) operations. </p>

<p>The Fourier convolution theorem requires the two arrays being convolved to be the same size, because the product in the theorem is a pointwise product.  Consequently, in order to convolve two arrays that are not the same size, in our case an NxN image and a 3x3 kernel, the smaller of the two arrays must be zero-padded to the size of the larger array before performing the FFT.  The FFT-based convolution must act on two N2 images instead of one N2 image and one 32 image. Thus, instead of getting the desired 9N2 to 2log2(3) Nlog2(N) speedup that we might have expected, we instead get 9N2 vs 2N2log2(N).  This actually means the FFT convolution will be a slower algorithm for any N>32. </p>

In our implementation, the FFT convolution method actually does somewhat worse than 2N2log2(N) due to the type conversions from the pixels’ native uint_8 to the 64-bit complex required by the FFT and the associated additional memory allocations.  Our experiments on 2.4 MP and larger images (see results section) confirm this.




## Results of running the program on flowers.jpg

<p> This shows the result of running the multiscale edge detection program on flowers.jpg with the following scales taken into account: <b>x1, x2, x4, x6, x8</b></p>

<p float="left" align="center">
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/flowers.jpg" width="192" height="123">
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/edges_1x.jpg" width="192" height="123">
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/edges_2x.jpg" width="192" height="123">
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/edges_4x.jpg" width="192" height="123">
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/edges_6x.jpg" width="192" height="123">
<img src="https://github.com/jimmykdunn/multiscaleEdgeDetection/blob/master/Photos%20for%20Readme/edges_8x.jpg" width="192" height="123">
</p>


## References

* Hat tip to anyone whose code was used
* Inspiration
* etc
