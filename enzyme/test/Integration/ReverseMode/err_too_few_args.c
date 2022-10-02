// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g -O0 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g -O1 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g -O2 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g -O3 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify; fi

extern void __enzyme_autodiff(void*);

void g(int size) {
}

void square() {
    __enzyme_autodiff((void*)g); // expected-error {{Enzyme: Insufficient number of args passed to derivative call required 1 primal args, found 0}}
}
