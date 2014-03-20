#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 256

// Lab4: Host Helper Functions (allocate your own data structure...)


// Lab4: Device Functions


// Lab4: Kernel Functions


__global__ void scan_workefficient(float *g_odata, float *g_idata, int n)
{
    // Dynamically allocated shared memory for scan kernels
    extern  __shared__  float temp[];

    int thid = threadIdx.x;

    int offset = 1;

    // Cache the computational window in shared memory
    temp[2*thid]   = g_idata[2*thid];
    temp[2*thid+1] = g_idata[2*thid+1];

    // build the sum in place up the tree
    for (int d = n>>1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)      
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    // scan back down the tree

    // clear the last element
    if (thid == 0)
    {
        temp[n - 1] = 0;
    }   

    // traverse down the tree building the scan in place
    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();

        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            float t   = temp[ai];
            temp[ai]  = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    // write results to global memory
    g_odata[2*thid]   = temp[2*thid];
    g_odata[2*thid+1] = temp[2*thid+1];
}


// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements)
{

	scan_workefficient<<<1,512,4096>>>(outArray, inArray,numElements);


}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
