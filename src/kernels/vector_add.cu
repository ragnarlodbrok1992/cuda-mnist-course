#include "vector_add.h"

__global__ void vector_add(const float* A, const float* B, float* C, size_t num_of_elements) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x; // TODO: Explain this

  if (i < num_of_elements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

void vector_add_host(size_t blocks, size_t threads,
    const float* A, const float* B, float* C,
    size_t num_of_elements) {
  vector_add<<<blocks, threads>>>(A, B, C, num_of_elements);
}


