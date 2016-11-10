#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h>

#include "info.h"
#include "err.h"

__global__ void matrix_mul(int *a, int *b, int *c) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int posx = threadIdx.x;
    int posy = threadIdx.y;

    int step = 0;

    for ( int w = 0; w < N/SHARED_BLOCK_SIZE; w++ ) {

        __shared__ int as[SHARED_BLOCK_SIZE * SHARED_BLOCK_SIZE];
        __shared__ int bs[SHARED_BLOCK_SIZE * SHARED_BLOCK_SIZE];

        as[posy * SHARED_BLOCK_SIZE + posx] = a[blockIdx.y * blockDim.x * N + w * blockDim.x     + posx + posy * N ];
        bs[posy * SHARED_BLOCK_SIZE + posx] = b[blockIdx.x * blockDim.x     + w * blockDim.x * N + posx + posy * N ];

        __syncthreads();

        for ( int i = 0; i  < SHARED_BLOCK_SIZE ; i ++ ) {
            step += as[posy * SHARED_BLOCK_SIZE + i   ] *
                    bs[i    * SHARED_BLOCK_SIZE + posx];
        }

        __syncthreads();
    }

    c[row * N + col] = step;
}

int main() {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N2 * sizeof(int);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuErrchk( cudaMalloc( (void**) &d_a, size));
    gpuErrchk( cudaMalloc( (void**) &d_b, size));
    gpuErrchk( cudaMalloc( (void**) &d_c, size));

    a = (int*) malloc ( size );
    b = (int*) malloc ( size );
    c = (int*) malloc ( size );

    if ( a == NULL ) { fprintf(stderr, "Failed to allocate a\n"); abort(); }
    if ( b == NULL ) { fprintf(stderr, "Failed to allocate b\n"); abort(); }
    if ( c == NULL ) { fprintf(stderr, "Failed to allocate c\n"); abort(); }

    for ( int i = 0; i  < N ; i ++ ) {
        for ( int j = 0; j  < N ; j ++ ) {
            a[i*N + j] = 0;
            b[i*N + j] = 0;
            c[i*N + j] = 0;
            if ( i == j ) {
                a[i*N + j] = 1;
                b[i*N + j] = 1;
                c[i*N + j] = 1;
            }
        }
    }

#ifdef __output
    printf("Matrix size = %d %d\n", N, N);
    printf("Number of elements = %d\n", N2);
    printf("Grid size = %d %d \n", NBLOCKS, NBLOCKS);
    printf("Shared size = %d %d \n", SHARED_BLOCK_SIZE, SHARED_BLOCK_SIZE);
    printf("Number of grid elements = %d\n", NBLOCKS * NBLOCKS);
    printf("Number of elements per grid = %d\n", THREADS_PER_BLOCK * THREADS_PER_BLOCK);
    printf("Threads per block = %d %d\n", THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    printf("Check %d = %d\n", THREADS_PER_BLOCK2 * NBLOCKS * NBLOCKS, N2);
#endif

    gpuErrchk( cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice ));
    gpuErrchk( cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice ));

    dim3 block  = dim3(NBLOCKS,
                       NBLOCKS,
                       1                  );
    dim3 thread = dim3(THREADS_PER_BLOCK,
                       THREADS_PER_BLOCK,
                       1                 );

    cudaEventRecord(start);
    matrix_mul<<< block, thread >>>(d_a, d_b, d_c);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    /*CudaCheckError();*/

    gpuErrchk( cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost ));

    for ( int i = 0; i  < N ; i ++ ) {
        for ( int j = 0; j  < N ; j ++ ) {
            if ( c[i*N + j] != 0  && i != j ) {
                fprintf(stderr, "Found nonzero outside the main diagonal\n");
                abort();
            } else if ( c[i*N + j] != 1  && i == j ) {
                fprintf(stderr, "Found something not 1 in the main diagonal\n");
                abort();
            }
            /*printf("%d", c[i*N + j]);*/
        }
        /*printf("\n");*/
    }
    /*printf("Matrix ok\n");*/

    printf("%d %f\n", N, milliseconds);

    free(a);
    free(b);
    free(c);
    gpuErrchk( cudaFree(d_a));
    gpuErrchk( cudaFree(d_b));
    gpuErrchk( cudaFree(d_c));

    return 0;
}
