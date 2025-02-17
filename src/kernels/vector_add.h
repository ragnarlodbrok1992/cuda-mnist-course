#ifndef _VECTOR_ADD_H_
#define _VECTOR_ADD_H_

void vector_add_host(size_t blocks, size_t threads, const float *A, const float *B, float *C, size_t num_of_elements);

#endif // _VECTOR_ADD_H_
