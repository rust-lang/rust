// RUN: %clang++ -I/usr/include/eigen3 -Xclang -new-struct-path-tbaa -mllvm -force-vector-width=1 -ffast-math -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -I/usr/include/eigen3 -Xclang -new-struct-path-tbaa -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -I/usr/include/eigen3 -Xclang -new-struct-path-tbaa -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -I/usr/include/eigen3 -Xclang -new-struct-path-tbaa -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions %O0TBAA %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli - 
// RUN: %clang++ -I/usr/include/eigen3 -Xclang -new-struct-path-tbaa -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang++ -I/usr/include/eigen3 -Xclang -new-struct-path-tbaa -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang++ -I/usr/include/eigen3 -Xclang -new-struct-path-tbaa -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang++ -I/usr/include/eigen3 -Xclang -new-struct-path-tbaa -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions %O0TBAA %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli - 

#define EIGEN_NO_AUTOMATIC_RESIZING 1
#define EIGEN_DONT_ALIGN 1
#define EIGEN_NO_DEBUG 1
#define EIGEN_UNROLLING_LIMIT 0
#define EIGEN_DONT_VECTORIZE 1

#include "test_utils.h"


void memcpy(float* __restrict dst, float* __restrict src, size_t count) {
    for(size_t i=0; i<count/sizeof(float); i++) {
        dst[i] = src[i];
    }
}

void memcpy(double* __restrict dst, double* __restrict src, size_t count) {
    for(size_t i=0; i<count/sizeof(double); i++) {
        dst[i] = src[i];
    }
}


template<typename T>
void memcpy(T* __restrict dst, T* __restrict src, size_t count) {
    for(size_t i=0; i<count/sizeof(T); i++) {
        dst[i] = src[i];
    }
}



#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::Tensor;

constexpr size_t IN = 4, OUT = 4, NUM = 5;


namespace Eigen {
namespace internal {

template<> struct smart_copy_helper<float, true> {
EIGEN_DEVICE_FUNC static inline void run(const float* start, const float* end, float* target) {
    for(unsigned i=0; start+i != end; i++) {
        target[i] = start[i];
    }
}
};
};

};


extern "C" {
    extern double __enzyme_autodiff(void*, const Tensor<float, 2>* __restrict K, const Tensor<float, 2>* __restrict Kp, const Tensor<float, 4>* __restrict I, const Tensor<float, 4>* __restrict Ip, Tensor<float, 4>* __restrict O, Tensor<float, 4>* __restrict Op);
}

__attribute__((noinline))
static void matvec(const Tensor<float, 2>* __restrict K, const Tensor<float, 4>* __restrict In, Tensor<float, 4>* Out) {
  Eigen::array<ptrdiff_t, 2> dims({1, 2});
  *Out = In->convolve(*K, dims);
}

int main(int argc, char** argv) {

    Tensor<float, 4> input(3, 3, 7, 11);
    Tensor<float, 2> kernel(2, 2);
    Tensor<float, 4> output(3, 2, 6, 11);
    input.setRandom();
    kernel.setRandom();

    Tensor<float, 4> inputp(3, 3, 7, 11);
    Tensor<float, 2> kernelp(2, 2);
    Tensor<float, 4> outputp(3, 2, 6, 11);
    inputp.setZero();
    kernelp.setZero();
    outputp.setRandom(); //One();
    
    Tensor<float, 2> expected_kernel(2, 2);
    expected_kernel.setZero();
for (int i = 0; i < 3; ++i) {
  for (int j = 0; j < 2; ++j) {
    for (int k = 0; k < 6; ++k) {
      for (int l = 0; l < 11; ++l) {
        const float result = output(i,j,k,l);
        const float expected = input(i,j+0,k+0,l) * kernel(0,0) +
                               input(i,j+1,k+0,l) * kernel(1,0) +
                               input(i,j+0,k+1,l) * kernel(0,1) +
                               input(i,j+1,k+1,l) * kernel(1,1);
        //VERIFY_IS_APPROX(result, expected);
        //VERIFY_IS_APPROX(result, expected);
		for(int si=0; si<2; si++)
		for(int sj=0; sj<2; sj++)
			expected_kernel(si,sj) += outputp(i, j, k, l) * input(i, j+si, k+sj, l);
      }
    }
  }
}

    matvec(&kernel, &input, &output);
    printf("did original\n");
    __enzyme_autodiff((void*)matvec, &kernel, &kernelp, &input, &inputp, &output, &outputp);
 

	for(int si=0; si<2; si++)
	for(int sj=0; sj<2; sj++) {
        fprintf(stderr, "kernelp(si=%d, sj=%d)=%f, expected_kernel(si=%d, sj=%d)=%f\n", si, sj, kernelp(si, sj), si, sj, expected_kernel(si, sj) );
        APPROX_EQ( kernelp(si, sj), expected_kernel(si, sj), 1e-3);
    }
     
}
