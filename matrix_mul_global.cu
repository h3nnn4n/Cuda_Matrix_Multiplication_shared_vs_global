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

    c[row * N + col] = 0;

    for ( int i = 0; i  < N ; i ++ ) {
        c[row * N + col] += a[row * N + i] * b[i * N + col];
    }
}

int main() {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N2 * sizeof(int);

    struct timeval timevalA;
    struct timeval timevalB;

    struct timeval timevalA2;
    struct timeval timevalB2;

    gettimeofday(&timevalA2,NULL);

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

    gettimeofday(&timevalA,NULL);
    matrix_mul<<< block, thread >>>(d_a, d_b, d_c);
    gettimeofday(&timevalB,NULL);


    gpuErrchk( cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost ));

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
    gpuErrchk( cudaFree(a));
    gpuErrchk( cudaFree(b));
    gpuErrchk( cudaFree(c));

    return 0;
}
