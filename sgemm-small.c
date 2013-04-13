#include <stdio.h>
#include <emmintrin.h>

void sgemm( int m, int n, int d, float *A, float *C )
{
    __m128 first;
    __m128 second;
    __m128 result;
    __m128 current;
    float *trans;
    float *c;
    for( int j = 0; j < n; j++ ) {
	for( int k = 0; k < m; k++ ) {
	    trans = &A[j*(n+1)+k*n];
	    second = _mm_load1_ps(trans);
	    for( int i = 0; i < n/8*8; i += 8) {
		first = _mm_loadu_ps(&A[i+k*n]);
		c = &C[i+j*(n)];
		result = _mm_mul_ps(first, second);
		_mm_storeu_ps(c, _mm_add_ps(_mm_loadu_ps(c), result));
		first = _mm_loadu_ps(&A[i+4+k*n]);
		c = &(C[i+4+j*(n)]);
		result = _mm_mul_ps(first, second);
		_mm_storeu_ps(c, _mm_add_ps(_mm_loadu_ps(c), result));
	    }
	    for (int i = n/8*8; i < n; i++) {
		C[i+j*n] += A[i+k*(n)] * *trans;
	    }
	}
    }
}
