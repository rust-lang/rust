// RUN: %clang++ -std=c++11 -fno-exceptions -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme-inline=1 -S | %lli -

#include <cmath>

template <typename T, int... n>
struct tensor;

template <typename T, int n>
struct tensor<T, n> {

  T& operator[](int i) { return value[0]; };

  T value[n];
};

template <typename T, int first, int... rest>
struct tensor<T, first, rest...> {

  tensor<T, rest...>& operator[](int i) { return value[0]; };

  tensor<T, rest...> value[first];
};

int enzyme_dup;
int enzyme_out;
int enzyme_const;

template<typename return_type, typename... Args>
return_type __enzyme_autodiff(Args...);

template<typename return_type, typename... Args>
return_type __enzyme_fwddiff(Args...);

extern "C" {
__attribute__((noinline))
constexpr double ptr(double* A) {
  return A[0];
}
}

template <int n>
__attribute__((noinline))
constexpr tensor<double, 1, 1> pdev(const tensor<double, 1, 1>& A) {
  auto devA = A;
  auto trA = ptr((double*)&A);
  devA[0][0] -= trA;
  devA[0][0] -= trA;
    devA[0][0] -= trA;
  return devA;
}

extern "C" {
    double mystress_calculation(void* __restrict__ D, const tensor<double, 1, 1> & __restrict__ du_dx) {
      auto devB = pdev<2>(du_dx);
      
      return 2 * devB[0][0];
    }
}

int main(int argc, char * argv[]) {
    	tensor<double, 1, 1> dudx = {{{0.0}}};
    	tensor<double, 1, 1> ddudxi = {{{0.0}}};
    
		tensor<double, 1, 1> gradient{};
		tensor<double, 1, 1> sigma{};
		tensor<double, 1, 1> dir{};

		dir[0][0] = 1;
        // Forward pass of gradient can segfault if forward and reverse preprocess
        // functions collide in cache.
		for (int i = 0; i < 2; i++)
		{

				__enzyme_autodiff<void>(mystress_calculation,
										&enzyme_const, nullptr,
										&enzyme_dup, &dudx, &gradient);
		}
        
		__enzyme_fwddiff<void>(mystress_calculation,
							   &enzyme_const, nullptr,
							   &enzyme_dup, &dudx, &ddudxi);
}
