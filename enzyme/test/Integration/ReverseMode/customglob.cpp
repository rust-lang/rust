// RUN: if [ %llvmver -ge 12 ]; then %clang++ -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -O0 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -O3 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi

#include "test_utils.h"

double __enzyme_autodiff(void*, ...);

enum class MyMemoryType
{
   DEFAULT
};

extern MyMemoryType host_mem_type;

__attribute__((noinline))
void* alloc(int size, MyMemoryType mt) {
    return malloc(size);
}

double square(double a)
{
  double* D = (double*)alloc(sizeof(double), host_mem_type);
  D[0] = a;
  return D[0];
}
void* __enzyme_inactive_global = &host_mem_type;

int main() {
  double out = __enzyme_autodiff((void*)square, 10.0);
  APPROX_EQ(out, 1.0, 1e-7);
}
