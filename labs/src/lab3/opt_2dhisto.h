#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t* input, size_t height, size_t width, uint32_t* bins);

/* Include below the function headers of any other functions that you implement */

void* AllocateDeviceArray(int size);

void MemsetHandler(uint32_t* bins);

void CopyToDeviceArray(void *device, const void *host, int size);

void CopyFromDeviceArray(uint32_t *host, const uint32_t *device, int size);

void FreeDeviceArray(void *device);

void FreeHostArray(void *host);

#endif
