#include <stdio.h>
#include <emmintrin.h>

void sgemm( int m, int n, int d, float *A, float *C )
{
    int i,j,k;
    __m128 first;
    __m128 first1;
    __m128 first2;
    __m128 first3;
    __m128 first4;
    __m128 first5;
    __m128 first6;
    __m128 first7;
    __m128 second;
    __m128 second1;
    __m128 second2;
    __m128 second3;
    __m128 second4;
    __m128 second5;
    __m128 second6;
    __m128 second7;
    __m128 result;
    __m128 result1;
    __m128 result2;
    __m128 result3;
    __m128 result4;
    __m128 result5;
    __m128 result6;
    __m128 result7;
    float *trans;
    float *a2;
    float *a;
    float *c;
    for( j = 0; j < n; j++ ) {
	trans = A+j*(n+1);
	for( k = 0; k < m/8*8; k += 8 ) {
	    second = _mm_load1_ps(trans+k*n);
	    second1 = _mm_load1_ps(trans+(k+1)*n);
	    second2 = _mm_load1_ps(trans+(k+2)*n);
	    second3 = _mm_load1_ps(trans+(k+3)*n);
	    second4 = _mm_load1_ps(trans+(k+4)*n);
	    second5 = _mm_load1_ps(trans+(k+5)*n);
	    second6 = _mm_load1_ps(trans+(k+6)*n);
	    second7 = _mm_load1_ps(trans+(k+7)*n);
	    for( i = 0; i < n/8*8; i += 8) {
		a = A+i+k*n;
		a2 = a;
		first = _mm_loadu_ps(a);
		a+=n;
		first1 = _mm_loadu_ps(a);
		a+=n;
		first2 = _mm_loadu_ps(a);
		a+=n;
		first3 = _mm_loadu_ps(a);
		a+=n;
		first4 = _mm_loadu_ps(a);
		a+=n;
		first5 = _mm_loadu_ps(a);
		a+=n;
		first6 = _mm_loadu_ps(a);
		a+=n;
		first7 = _mm_loadu_ps(a);
		c = C+i+j*n;
		result = _mm_mul_ps(first, second);
		result1 = _mm_mul_ps(first1, second1);
		result2 = _mm_mul_ps(first2, second2);
		result3 = _mm_mul_ps(first3, second3);
		result4 = _mm_mul_ps(first4, second4);
		result5 = _mm_mul_ps(first5, second5);
		result6 = _mm_mul_ps(first6, second6);
		result7 = _mm_mul_ps(first7, second7);
		result = _mm_add_ps(result, result1);
		result = _mm_add_ps(result, result2);
		result = _mm_add_ps(result, result3);
		result = _mm_add_ps(result, result4);
		result = _mm_add_ps(result, result5);
		result = _mm_add_ps(result, result6);
		result = _mm_add_ps(result, result7);
		_mm_storeu_ps(c, _mm_add_ps(_mm_loadu_ps(c), result));
		a = a2+4;
		first = _mm_loadu_ps(a);
		a+=n;
		first1 = _mm_loadu_ps(a);
		a+=n;
		first2 = _mm_loadu_ps(a);
		a+=n;
		first3 = _mm_loadu_ps(a);
		a+=n;
		first4 = _mm_loadu_ps(a);
		a+=n;
		first5 = _mm_loadu_ps(a);
		a+=n;
		first6 = _mm_loadu_ps(a);
		a+=n;
		first7 = _mm_loadu_ps(a);
 		c += 4;
		result = _mm_mul_ps(first, second);
		result1 = _mm_mul_ps(first1, second1);
		result2 = _mm_mul_ps(first2, second2);
		result3 = _mm_mul_ps(first3, second3);
		result4 = _mm_mul_ps(first4, second4);
		result5 = _mm_mul_ps(first5, second5);
		result6 = _mm_mul_ps(first6, second6);
		result7 = _mm_mul_ps(first7, second7);
		result = _mm_add_ps(result, result1);
		result = _mm_add_ps(result, result2);
		result = _mm_add_ps(result, result3);
		result = _mm_add_ps(result, result4);
		result = _mm_add_ps(result, result5);
		result = _mm_add_ps(result, result6);
		result = _mm_add_ps(result, result7);
		_mm_storeu_ps(c, _mm_add_ps(_mm_loadu_ps(c), result));
	    }
	    for (i = n/8*8; i < n; i++) {
		C[i+j*n] += (A[i+k*(n)] * *(trans + k*n)) + (A[i+(k+1)*n] * *(trans + (k+1)*n))
		    + (A[i+(k+2)*n] * *(trans + (k+2)*n)) + (A[i+(k+3)*n] * *(trans + (k+3)*n))
		    + (A[i+(k+4)*(n)] * *(trans + (k+4)*n)) + (A[i+(k+5)*n] * *(trans + (k+5)*n))
		    + (A[i+(k+6)*n] * *(trans + (k+6)*n)) + (A[i+(k+7)*n] * *(trans + (k+7)*n));
	    }
	}
	for (k = m/8*8; k < m; k++) {
	    second = _mm_load1_ps(trans+k*n);
	    for (i = 0; i < n/8*8; i += 8) {
		a = A+i+k*n;
		first = _mm_loadu_ps(a);
		c = C+i+j*n;
		result = _mm_mul_ps(first, second);
		_mm_storeu_ps(c, _mm_add_ps(_mm_loadu_ps(c), result));
		a+=4;
		c+=4;
		first = _mm_loadu_ps(a);
		result = _mm_mul_ps(first, second);
		_mm_storeu_ps(c, _mm_add_ps(_mm_loadu_ps(c), result));
	    }
	    for (i = n/8*8; i < n; i++) {
		C[i+j*n] += A[i+k*n] * A[j*(n+1)+k*n];
	    }
	}
    }
}
