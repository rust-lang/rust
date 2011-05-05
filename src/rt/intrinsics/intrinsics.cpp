// Use `clang++ -emit-llvm -S -arch i386 -O3 -I../isaac -I../uthash -o
//      intrinsics.ll intrinsics.cpp`

#include "../rust_internal.h"

extern "C" size_t
rust_intrinsic_vec_len(rust_task *task, type_desc *ty, rust_vec *v)
{
    return v->fill / ty->size;
}

