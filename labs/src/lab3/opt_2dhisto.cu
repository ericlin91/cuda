//
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"


__global__ void opt_2dhistoKernel(uint32_t *input, uint32_t* bins, size_t input_height, size_t input_width);
__global__ void saturated(uint32_t *bins);

void opt_2dhisto(uint32_t* input, size_t height, size_t width, uint32_t* bins)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */

    cudaMemset  (bins, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));

    //  int threads = 1024;
    // opt_2dhistoKernel<<<height*width/threads, threads>>>(input, bins, height, width);
    
    // OPTIMIZATION 1 AND 2 GRID
    //opt_2dhistoKernel<<<height,1>>>(input, bins, height, width);
    
    // OPTIMIZATION 3 GRID
    //opt_2dhistoKernel<<<height,32>>>(input, bins, height, width);

    // OPTIMIZATION 4 GRID
    //need gridDim to be multiple 4096
    //need blockDim to be multiple of 3984
    opt_2dhistoKernel<<<4096,83>>>(input, bins, height, width);

    saturated<<<1,1024>>>(bins);

    cudaThreadSynchronize();
}


//__device__ uint counter = 0;

/* Include below the implementation of any other functions you need */
__global__ void opt_2dhistoKernel(uint32_t *input, uint32_t* bins, size_t input_height, size_t input_width){


/* OPTIMIZATION 1		
	size_t j = blockIdx.x;
        for (size_t i = 0; i < input_width; ++i)
        {           

            // Increment the appropriate bin, but do not roll-over the max value
		atomicAdd(&bins[input[j*input_height+i]],1);
        }
*/

 // OPTIMIZATION 2
/*
    __shared__ uint32_t s_Hist[HISTO_WIDTH];

    for (int pos = 0; pos < HISTO_WIDTH; pos++){
        s_Hist[pos] = 0;
    }

    size_t j = blockIdx.x;
    for (size_t i = 0; i < input_width; ++i)
    {  
        // Increment the appropriate bin, but do not roll-over the max value
        atomicAdd(&s_Hist[input[j*input_height+i]],1);
    }

    for (int pos = 0; pos < HISTO_WIDTH; pos++){
        atomicAdd(bins+pos, s_Hist[pos]);
    }
*/

 //OPTIMIZATION 3
/*    
    __shared__ uint32_t s_Hist[HISTO_WIDTH];

    for (int pos = threadIdx.x; pos < HISTO_WIDTH; pos += blockDim.x){
        s_Hist[pos] = 0;
    }

    __syncthreads();

    size_t j = blockIdx.x;
    for (size_t i = threadIdx.x; i < input_width; i += blockDim.x)
    {  
        // Increment the appropriate bin, but do not roll-over the max value
        atomicAdd(&s_Hist[input[j*input_height+i]],1);
    }

    __syncthreads();

    for (int pos = threadIdx.x; pos < HISTO_WIDTH; pos += blockDim.x){
        atomicAdd(bins+pos, s_Hist[pos]);
    }
*/

//OPTIMIZATION 4

        // Initialize shared memory
        __shared__ uint32_t s_bins[HISTO_WIDTH];
        for (int ii = threadIdx.x; ii<HISTO_WIDTH ; ii+=blockDim.x) s_bins[ii] = 0;
        __syncthreads ();

        // Atomic add to shared memory
     	for (int j = blockIdx.x; j < input_height;j += gridDim.x){
                for (int i = threadIdx.x; i < input_width;i += blockDim.x){
                       	atomicAdd(s_bins + input[j*input_height+i],1);//shared write
                         // atomicAdd(bins + input[j*input_height+i],1);//global write
                }
        }
        __syncthreads ();

        //Atomic add shared memory to global
        for (int ii = threadIdx.x; ii<HISTO_WIDTH ; ii+=blockDim.x)
            atomicAdd(&bins[ii], s_bins[ii]);



}


__global__ void saturated(uint32_t *bins){
 	int tid = threadIdx.x;
	if (bins[tid]>UINT8_MAX) bins[tid]=UINT8_MAX;
}

// Allocate a device array 
void* AllocateDeviceArray(int size){
    void *ret;
    cudaMalloc(&ret, size);
    return ret;
}

void MemsetHandler(uint32_t* bins){
    cudaMemset  (bins, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));
}

// Copy a host array to a device array.
void CopyToDeviceArray(void *device, const void *host, int size)
{
    cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
}

// Copy a device array to a host array.
void CopyFromDeviceArray(uint32_t *host, const uint32_t *device, int size)
{
    cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
}

// Free a device array.
void FreeDeviceArray(void *device)
{
    cudaFree(device);
}

// Free a host array
void FreeHostArray(void *host)
{
    free(host);
}
