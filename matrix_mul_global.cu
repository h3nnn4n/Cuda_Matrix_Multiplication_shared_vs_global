#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include<sys/time.h>
#include<time.h>

#define N  (256)
#define N2 (N*N)
#define NBLOCKS 8
#define THREADS_PER_BLOCK  ((N/NBLOCKS))
#define THREADS_PER_BLOCK2 ((N/NBLOCKS)*(N/NBLOCKS))

__global__ void vector_add(int *a, int *b, int *c) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    c[blockDim.x * blockIdx.y + blockIdx.x] = 0;
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
            } }
    }

#ifdef __output
    printf("Matrix size = %d %d\n", N, N);
    printf("Number of elements = %d\n", N2);
    printf("Grid size = %d %d \n", NBLOCKS, NBLOCKS);
    printf("Number of grid elements = %d\n", NBLOCKS * NBLOCKS);
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
    vector_add<<< block, thread >>>(d_a, d_b, d_c);
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
