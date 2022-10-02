// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g -O0 %s -S -emit-llvm -o /dev/null %loadClangEnzyme -Xclang -verify -Rpass=enzyme -mllvm -enzyme-postopt=0; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g -O1 %s -S -emit-llvm -o /dev/null %loadClangEnzyme -Xclang -verify -Rpass=enzyme -mllvm -enzyme-postopt=0; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g -O2 %s -S -emit-llvm -o /dev/null %loadClangEnzyme -Xclang -verify -Rpass=enzyme -mllvm -enzyme-postopt=0; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g -O3 %s -S -emit-llvm -o /dev/null %loadClangEnzyme -Xclang -verify -Rpass=enzyme -mllvm -enzyme-postopt=0; fi

// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g0 -O0 %s -S -emit-llvm -o /dev/null %loadClangEnzyme -Xclang -verify -Rpass=enzyme -mllvm -enzyme-postopt=0; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g0 -O1 %s -S -emit-llvm -o /dev/null %loadClangEnzyme -Xclang -verify -Rpass=enzyme -mllvm -enzyme-postopt=0; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g0 -O2 %s -S -emit-llvm -o /dev/null %loadClangEnzyme -Xclang -verify -Rpass=enzyme -mllvm -enzyme-postopt=0; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g0 -O3 %s -S -emit-llvm -o /dev/null %loadClangEnzyme -Xclang -verify -Rpass=enzyme -mllvm -enzyme-postopt=0; fi

extern void __enzyme_autodiff(void*, ...);

void g(double* in, int N) {

    for (int i=0; i<N; i++) {
        double load = in[i & 1]; // expected-remark {{Load may need caching}}, expected-remark {{Load must be recomputed}}, expected-remark {{Caching instruction}}
        double sq = load * load;
        in[i & 1] = sq;
    }
}

void square(double* x, double* dx) {
    __enzyme_autodiff((void*)g, x, dx, 20); 
}
