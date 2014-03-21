#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 1024 //256
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
// Lab4: Host Helper Functions (allocate your own data structure...)


// Lab4: Device Functions


// Lab4: Kernel Functions


__global__ void scan_workefficient(float *g_odata, float *g_idata, float *g_sums, int n, int blockID_offset)
{
    // Dynamically allocated shared memory for scan kernels
    extern  __shared__  float temp[];

    int thid = threadIdx.x;

    int bid = blockIdx.x + blockID_offset;

    int offset = 1;

    // Cache the computational window in shared memory
    int block_offset = BLOCK_SIZE*bid;
    temp[2*thid]   = g_idata[2*(thid+block_offset)];
    temp[2*thid+1] = g_idata[2*(thid+block_offset)+1];

    int ai = thid;
    int bi = thid + (n/2);

    // // compute spacing to avoid bank conflicts
    // int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    // int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // // Cache the computational window in shared memory
    // temp[ai + bankOffsetA] = g_idata[ai]; 
    // temp[bi + bankOffsetB] = g_idata[bi]; 

    // build the sum in place up the tree
    for (int d = n>>1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)      
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            // ai += ai/NUM_BANKS;
            // bi += bi/NUM_BANKS;

            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    // scan back down the tree

    // clear the last element
    if (thid == 0)
    {
    	if(g_sums){
    		//g_sums[0] = 0;
    		g_sums[bid] = temp[n - 1];
    	}
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

            // ai += ai/NUM_BANKS;
            // bi += bi/NUM_BANKS;

            float t   = temp[ai];
            temp[ai]  = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    // write results to global memory
    g_odata[2*(thid+block_offset)]   = temp[2*thid];
    g_odata[2*(thid+block_offset)+1] = temp[2*thid+1];
}

__global__ void consolidate(float *g_odata, float *g_sums)
{
	int thid = threadIdx.x;


    // write results to global memory
    g_odata[2*(thid+BLOCK_SIZE)]   += g_sums[0];
    g_odata[2*(thid+BLOCK_SIZE)+1] += g_sums[0];
    // g_odata[2*(thid+BLOCK_SIZE)]   += g_incr[1];
    // g_odata[2*(thid+BLOCK_SIZE)+1] += g_incr[1];
}

__global__ void update(float *g_odata, float *g_incr, int blockID_offset)
{
    int thid = threadIdx.x;

    int bid = blockIdx.x + blockID_offset;

    // Cache the computational window in shared memory
    int block_offset = BLOCK_SIZE*bid;
    g_odata[2*(thid+block_offset)] += g_incr[bid];
    g_odata[2*(thid+block_offset)+1] += g_incr[bid];

}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, float *sums, float *incr, float *incr_sums, float *incr_incr, int numElements)
{
	int num_blocks = numElements/(BLOCK_SIZE*2);
	if(num_blocks==0){
		num_blocks = 1;
	}
    if(num_blocks>2048){
        num_blocks=2048;
    }

	//first scan individual blocks
	scan_workefficient<<<num_blocks,BLOCK_SIZE,8192>>>(outArray, inArray, sums, BLOCK_SIZE*2, 0);
    scan_workefficient<<<num_blocks,BLOCK_SIZE,8192>>>(outArray, inArray, sums, BLOCK_SIZE*2, 2048);
    scan_workefficient<<<num_blocks,BLOCK_SIZE,8192>>>(outArray, inArray, sums, BLOCK_SIZE*2, 4096);
    scan_workefficient<<<num_blocks,BLOCK_SIZE,8192>>>(outArray, inArray, sums, BLOCK_SIZE*2, 6144);

    //at this point, sums is ready to be scanned
    scan_workefficient<<<4,BLOCK_SIZE,8192>>>(incr, sums, incr_sums, num_blocks, 0);
    scan_workefficient<<<1,BLOCK_SIZE,8192>>>(incr_incr, incr_sums, NULL, 4, 0);
    update<<<4, BLOCK_SIZE, 8192>>>(incr, incr_incr, 0);

	update<<<num_blocks, BLOCK_SIZE, 8192>>>(outArray, incr, 0);
    update<<<num_blocks, BLOCK_SIZE, 8192>>>(outArray, incr, 2048);
    update<<<num_blocks, BLOCK_SIZE, 8192>>>(outArray, incr, 4096);
    update<<<num_blocks, BLOCK_SIZE, 8192>>>(outArray, incr, 6144);


}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
