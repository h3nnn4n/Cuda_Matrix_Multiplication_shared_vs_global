#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h>

#include "info.h"

__global__ void matrix_mul(int *a, int *b, int *c) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int posx = threadIdx.x;
    int posy = threadIdx.y;

    c[row * N + col] = 0;

    int step = 0;

    for ( int w = 0; w < N/SHARED_BLOCK_SIZE; w++ ) {

        __shared__ int as[SHARED_BLOCK_SIZE * SHARED_BLOCK_SIZE];
        __shared__ int bs[SHARED_BLOCK_SIZE * SHARED_BLOCK_SIZE];

        as[posy * N + posx] = a[row * N + col];
        bs[posy * N + posx] = b[row * N + col];

        __syncthreads();

        for ( int i = 0; i  < N ; i ++ ) {
            step += as[posy * N + i] * bs[i * N + posx];
        }

        __syncthreads();
    }

    c[row * N + col] = step;
}

int main() {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N2 * sizeof(int);

    struct timeval timevalA;
    struct timeval timevalB;

    struct timeval timevalA2; struct timeval timevalB2;

    gettimeofday(&timevalA2,NULL);

    cudaMalloc( (void**) &d_a, size);
    cudaMalloc( (void**) &d_b, size);
    cudaMalloc( (void**) &d_c, size);

    a = (int*) malloc ( size );
    b = (int*) malloc ( size );
    c = (int*) malloc ( size );

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
    printf("Number of grid elements = %d\n", NBLOCKS * NBLOCKS);
    printf("Number of elements per grid = %d\n", THREADS_PER_BLOCK * THREADS_PER_BLOCK);
    printf("Threads per block = %d %d\n", THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    printf("Check %d = %d\n", THREADS_PER_BLOCK2 * NBLOCKS * NBLOCKS, N2);
#endif

    cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
    cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice );

    dim3 block  = dim3(NBLOCKS,
                       NBLOCKS,
                       1                  );
    dim3 thread = dim3(THREADS_PER_BLOCK,
                       THREADS_PER_BLOCK,
                       1                 );

    gettimeofday(&timevalA,NULL);
    matrix_mul<<< block, thread >>>(d_a, d_b, d_c);
    gettimeofday(&timevalB,NULL);


    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost );

    gettimeofday(&timevalB2,NULL);

    /*for ( int i = 0; i  < N ; i ++ ) {*/
        /*for ( int j = 0; j  < N ; j ++ ) {*/
            /*printf("%d ", c[i*N + j]);*/
        /*}*/
        /*printf("\n");*/
    /*}*/

    printf("%d %f %f\n", N, timevalB.tv_sec-timevalA.tv_sec+(timevalB.tv_usec-timevalA.tv_usec)/(double)1000000,
                            timevalB2.tv_sec-timevalA2.tv_sec+(timevalB2.tv_usec-timevalA2.tv_usec)/(double)1000000
          );

    free(a);
    free(b);
    free(c);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
