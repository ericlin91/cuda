/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>

// includes, kernels
#include <scan_largearray_kernel.cu>  

#define DEFAULT_NUM_ELEMENTS 16777216 
#define MAX_RAND 3


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

int ReadFile(float*, char* file_name, int size);
void WriteFile(float*, char* file_name, int size);

extern "C" 
unsigned int compare( const float* reference, const float* data, 
                     const unsigned int len);
extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a scan test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    int errorM = 0;
    float device_time;
    float host_time;
    int* size = NULL; //(int*)malloc(1 * sizeof(int));
    unsigned int data2read = 1;
    int num_elements = 0; // Must support large, non-power-of-2 arrays

    // allocate host memory to store the input data
    unsigned int mem_size = sizeof( float) * num_elements;
    float* h_data = (float*) malloc( mem_size);

    //real_mem_size is the memsize rounded to the next highest power of two
    //this is so we can memset to 0 later to enable non-power of two arrays
    int exponent = (int)ceil(log2 ((double)DEFAULT_NUM_ELEMENTS));
    int real_mem_size = (int)pow(2, exponent) * sizeof(float);

    // * No arguments: Randomly generate input data and compare against the 
    //   host's result.
    // * One argument: Randomly generate input data and write the result to
    //   file name specified by first argument
    // * Two arguments: Read the first argument which indicate the size of the array,
    //   randomly generate input data and write the input data
    //   to the second argument. (for generating random input data)
    // * Three arguments: Read the first file which indicate the size of the array,
    //   then input data from the file name specified by 2nd argument and write the
    //   SCAN output to file name specified by the 3rd argument.
    switch(argc-1)
    {      
        case 2: 
            // Determine size of array
            cutReadFilei(argv[1], &size, &data2read, true);
            if(data2read != 1){
                printf("Error reading parameter file\n");
                exit(1);
            }

            num_elements = size[0];

            // allocate host memory to store the input data
            mem_size = sizeof( float) * num_elements;
            h_data = (float*) malloc( mem_size);

            for( unsigned int i = 0; i < num_elements; ++i) 
            {
                h_data[i] = (int)(rand() % MAX_RAND);
            }
            WriteFile(h_data, argv[2], num_elements);
        break;
    
        case 3:  // Three Arguments
            cutReadFilei(argv[1], &size, &data2read, true);
            if(data2read != 1){
                printf("Error reading parameter file\n");
                exit(1);
            }

            num_elements = size[0];
            
            // allocate host memory to store the input data
            mem_size = sizeof( float) * num_elements;
            h_data = (float*) malloc( mem_size);

            errorM = ReadFile(h_data, argv[2], size[0]);
            if(errorM != 1)
            {
                printf("Error reading input file!\n");
                exit(1);
            }
        break;
        
        default:  // No Arguments or one argument
            // initialize the input data on the host to be integer values
            // between 0 and 1000
            // Use DEFAULT_NUM_ELEMENTS num_elements
            num_elements = DEFAULT_NUM_ELEMENTS;
            
            // allocate host memory to store the input data
            mem_size = sizeof( float) * num_elements;
            h_data = (float*) malloc( mem_size);

            // initialize the input data on the host
            for( unsigned int i = 0; i < num_elements; ++i) 
            {
//                h_data[i] = 1.0f;
                h_data[i] = (int)(rand() % MAX_RAND);
            }
        break;  
    }    

    
    unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));

      
    // compute reference solution
    float* reference = (float*) malloc( mem_size);  
	cutStartTimer(timer);
    computeGold( reference, h_data, num_elements);
	cutStopTimer(timer);
    printf("\n\n**===-------------------------------------------------===**\n");
    printf("Processing %d elements...\n", num_elements);
    printf("Host CPU Processing time: %f (ms)\n", cutGetTimerValue(timer));
    host_time = cutGetTimerValue(timer);
    CUT_SAFE_CALL(cutDeleteTimer(timer));

    // **===-------- Lab4: Allocate data structure here -----------===**
    // allocate device memory input and output arrays
    float* d_idata = NULL;
    float* d_odata = NULL;
    float* sums = NULL;
    float* incr = NULL;
    float* incr_sums = NULL;
    float* incr_incr = NULL;


    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, real_mem_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, real_mem_size));

    //the consolidation arrays
    int num_blocks_needed = DEFAULT_NUM_ELEMENTS/(BLOCK_SIZE*2); //plus one is for leading zero so we have exclusive scan 
    CUDA_SAFE_CALL( cudaMalloc( (void**) &sums, sizeof(float)*num_blocks_needed));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &incr, sizeof(float)*num_blocks_needed));

    CUDA_SAFE_CALL( cudaMalloc( (void**) &incr_sums, sizeof(float)*20));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &incr_incr, sizeof(float)*20));
    
    //memset crucial to enable non power of 2 arrays
    CUDA_SAFE_CALL( cudaMemset( d_idata, 0, real_mem_size));
    CUDA_SAFE_CALL( cudaMemset( d_odata, 0, real_mem_size));

    CUDA_SAFE_CALL( cudaMemset( sums, 0, sizeof(float)*num_blocks_needed));
    CUDA_SAFE_CALL( cudaMemset( incr, 0, sizeof(float)*num_blocks_needed));

    // copy host memory to device input array
    CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice) );
    // initialize all the other device arrays to be safe
    CUDA_SAFE_CALL( cudaMemcpy( d_odata, h_data, mem_size, cudaMemcpyHostToDevice) );



    CUDA_SAFE_CALL( cudaMemset( incr_sums, 0, sizeof(float)*20));
    CUDA_SAFE_CALL( cudaMemset( incr_incr, 0, sizeof(float)*20));

    // **===-----------------------------------------------------------===**

    // Run just once to remove startup overhead for more accurate performance 
    // measurement
    prescanArray(d_odata, d_idata, sums, incr, incr_sums, incr_incr, 16);

    // Run the prescan
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    cutStartTimer(timer);
    
    // **===-------- Lab4: Modify the body of this function -----------===**
    prescanArray(d_odata, d_idata, sums, incr, incr_sums, incr_incr, num_elements);
    // **===-----------------------------------------------------------===**
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    cutStopTimer(timer);
    printf("CUDA Processing time: %f (ms)\n", cutGetTimerValue(timer));
    device_time = cutGetTimerValue(timer);
    printf("Speedup: %fX\n", host_time/device_time);

    // **===-------- Lab4: Deallocate data structure here -----------===**
    // deallocBlockSums();
    // **===-----------------------------------------------------------===**


    // copy result from device to host
    CUDA_SAFE_CALL(cudaMemcpy( h_data, d_odata, sizeof(float) * num_elements, 
                               cudaMemcpyDeviceToHost));

    float* h_sums = (float*) malloc( sizeof(float)*num_blocks_needed);
    float* h_incr = (float*) malloc( sizeof(float)*num_blocks_needed);

    CUDA_SAFE_CALL(cudaMemcpy( h_sums, sums, sizeof(float) * num_blocks_needed, 
                               cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy( h_incr, incr, sizeof(float) * num_blocks_needed, 
                               cudaMemcpyDeviceToHost));

    if ((argc - 1) == 3)  // Three Arguments, write result to file
    {
        WriteFile(h_data, argv[3], num_elements);
    }
    else if ((argc - 1) == 1)  // One Argument, write result to file
    {
        WriteFile(h_data, argv[1], num_elements);
    }

    //debug
    int count = 0;
    // int count_cuda = 0;
    // for (int i=0; i<4096;i +=1){

    //     count += reference[i];
    //     count_cuda += h_data[i];

    //     if (reference[i]==h_data[i])
    //      printf("%i  gold %f    cuda %f \n",i,reference[i],h_data[i]);
    //     // else
    //     //  printf("%i  gold %f    cuda %f \n",i,reference[i],h_data[i]);
    // }
    // for (int i=0; i<4096;i +=1){


    //      printf("%i  sum %f    incr %f \n",i,h_sums[i],h_incr[i]);
    //     // else
    //     //  printf("%i  gold %f    cuda %f \n",i,reference[i],h_data[i]);
    // }

    // Check if the result is equivalent to the expected soluion
    unsigned int result_regtest = cutComparef( reference, h_data, num_elements);
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");



    // cleanup memory
    cutDeleteTimer(timer);
    free( h_data);
    free( reference);
    cudaFree( d_odata);
    cudaFree( d_idata);
}


int ReadFile(float* M, char* file_name, int size)
{
	unsigned int elements_read = size;
	if (cutReadFilef(file_name, &M, &elements_read, true))
        return 1;
    else
        return 0;
}

void WriteFile(float* M, char* file_name, int size)
{
    cutWriteFilef(file_name, M, size, 0.0001f);
}

