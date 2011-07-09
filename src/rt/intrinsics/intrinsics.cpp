// Use `clang++ -emit-llvm -S -arch i386 -O3 -I../isaac -I../uthash
//      -I../arch/i386 -fno-stack-protector -o intrinsics.ll intrinsics.cpp`

#include "../rust_internal.h"

extern "C" size_t
rust_intrinsic_vec_len(rust_task *task, type_desc *ty, rust_vec *v)
{
    return v->fill / ty->size;
}

extern "C" size_t
rust_intrinsic_ivec_len(rust_task *task, type_desc *ty, rust_ivec *v)
{
    size_t fill;
    if (v->fill)
        fill = v->fill;
    else if (v->payload.ptr)
        fill = v->payload.ptr->fill;
    else
        fill = 0;
    return fill / ty->size;
}

extern "C" void *
rust_intrinsic_ptr_offset(rust_task *task, type_desc *ty, void *ptr,
                          uintptr_t count)
{
    return &((uint8_t *)ptr)[ty->size * count];
}

