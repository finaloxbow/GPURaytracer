#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720
#define NUM_CHANNELS 4

#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)
inline void checkCuda(cudaError_t result, char* func, char* file, int line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
			<< file << ":" << line << " '" << func << "\n";

		cudaDeviceReset();
		exit(99);
	}
}

class CudaManaged {
public:
	void* operator new(size_t len) {
		void* ptr;
		checkCudaErrors(cudaMallocManaged(&ptr, len));
		checkCudaErrors(cudaDeviceSynchronize());
		return ptr;
	}

	void operator delete(void* ptr) {
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaFree(ptr));
	}
};