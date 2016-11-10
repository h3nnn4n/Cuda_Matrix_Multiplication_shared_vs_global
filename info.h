#ifndef __infoooo
#define __infoooo

#define N  (256)
#define N2 (N*N)
#define NBLOCKS 8
#define SHARED_BLOCK_SIZE  4
#define THREADS_PER_BLOCK  ((N/NBLOCKS))
#define THREADS_PER_BLOCK2 ((N/NBLOCKS)*(N/NBLOCKS))

#endif /*__infoooo */
