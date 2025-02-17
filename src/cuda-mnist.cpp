#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels\vector_add.h"

#define VECTOR_SIZE 100000

int main(int argc, char* argv[]) {
  std::cout << "CUDA - Vector addition application." << std::endl;

  // Error code - checking return values from CUDA calls
  cudaError_t err = cudaSuccess;

  // Printing vector size
  size_t size = VECTOR_SIZE * sizeof(float);
  std::cout << "[Vector addition of " << VECTOR_SIZE <<
    " elements of type byte-width: " << sizeof(float)
    << " ]" << std::endl;

  // Allocating host memory
  // Vector A
  float* h_A = (float *)malloc(size);

  // Vector B
  float* h_B = (float *)malloc(size);

  // Vector C
  float* h_C = (float *)malloc(size);
  
  // Verifing allocation
  if (h_A == nullptr || h_B == nullptr || h_C == nullptr) {
    std::cerr << "Failed to allocate memory for host vectors!" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Initializing the host vectors
  for (size_t i = 0; i < VECTOR_SIZE; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Allocating CUDA device memory for vectors A, B and C
  // Vector A
  float* d_A = nullptr;
  err = cudaMalloc((void **)&d_A, size);
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device vector A! Error code --> " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  
  // Vector B
  float* d_B = nullptr;
  err = cudaMalloc((void **)&d_B, size);
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device vector B! Error code --> " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  
  // Vector C
  float* d_C = nullptr;
  err = cudaMalloc((void **)&d_C, size);
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device vector C! Error code --> " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }

  // Copy host vectors A and B from host memory to device memory
  std::cout << "Copying vectors A, B: host --> device." << std::endl;
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    std::cerr << "Failed to copy vector A from host to device! Error code --> " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    std::cerr << "Failed to copy vector B from host to device! Error code --> " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }

  // Launching vector add CUDA kernel
  size_t threads_per_block = 256;
  size_t blocks_per_grid = (VECTOR_SIZE + threads_per_block - 1) / threads_per_block;

  std::cout << "CUDA kernel launch with " << blocks_per_grid << " blocks of " << threads_per_block << " threads." << std::endl;

  // CUDA KERNEL CALL
  vector_add_host(blocks_per_grid, threads_per_block,
      d_A, d_B, d_C, VECTOR_SIZE);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    std::cout << "Failed to launch vector_add kernel! Error code --> " << cudaGetErrorString(err);
    exit(EXIT_FAILURE);
  }

  // Copy the device result vector in device memory to the host result vector in host memory.
  std::cout << "Copying vector C from device memory to host memory..." << std::endl;

  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    std::cout << "Failed to copy vector C from device to host! Error code --> " << cudaGetErrorString(err);
    exit(EXIT_FAILURE);
  }

  // Verify that the result vector is correct
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      std::cerr << "Result verification failed at element --> " << i << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  std::cout << "TEST passed!" << std::endl;

  // Free device global memory
  err = cudaFree(d_A);

  if (err != cudaSuccess) {
    std::cerr << "Failed to free device vector A! Error code --> " << cudaGetErrorString(err);
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_B);

  if (err != cudaSuccess) {
    std::cerr << "Failed to free device vector B! Error code --> " << cudaGetErrorString(err);
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_C);

  if (err != cudaSuccess) {
    std::cerr << "Failed to free device vector C! Error code --> " << cudaGetErrorString(err);
    exit(EXIT_FAILURE);
  }

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  std::cout << "Done, exiting..." << std::endl;

  return 0;
}
