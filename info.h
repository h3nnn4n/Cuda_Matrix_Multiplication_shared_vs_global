#ifndef __infoooo
#define __infoooo

#if 1

#ifndef M
#define M
#define N  ((512))
#else
#define N  ((M))
#endif

//#define N  ((512*2))
#define N2 (N*N)
#define NBLOCKS 32
#define SHARED_BLOCK_SIZE  ((N/NBLOCKS))
#define THREADS_PER_BLOCK  ((N/NBLOCKS))
#define THREADS_PER_BLOCK2 ((N/NBLOCKS)*(N/NBLOCKS))
#else

#ifndef M
#define M
#define N  ((512))
#else
#define N  ((M))
#endif

#define N2 (N*N)

#define THREADS_PER_BLOCK  (16)
#define NBLOCKS ((N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK)
#define SHARED_BLOCK_SIZE  ((N/NBLOCKS))
#define THREADS_PER_BLOCK2 ((THREADS_PER_BLOCK)*(THREADS_PER_BLOCK))
#endif

#if SHARED_BLOCK_SIZE > THREADS_PER_BLOCK
#error Shared block size must be smaller than the Threads block
#endif

#endif /*__infoooo */
