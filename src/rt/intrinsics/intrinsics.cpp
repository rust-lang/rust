// Use `clang++ -emit-llvm -S -arch i386 -O3 -I../isaac -I../uthash
//      -I../arch/i386 -fno-stack-protector -o intrinsics.ll intrinsics.cpp`

#include "../rust_internal.h"
#include <cstdlib>
#include <cstring>

extern "C" CDECL void
upcall_fail(rust_task *task, char const *expr, char const *file, size_t line);

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

extern "C" void
rust_intrinsic_vec_len_2(rust_task *task, size_t *retptr, type_desc *ty,
                         rust_vec *v)
{
    *retptr = v->fill / ty->size;
}

extern "C" void
rust_intrinsic_ivec_len_2(rust_task *task, size_t *retptr, type_desc *ty,
                          rust_ivec *v)
{
    size_t fill;
    if (v->fill)
        fill = v->fill;
    else if (v->payload.ptr)
        fill = v->payload.ptr->fill;
    else
        fill = 0;
    *retptr = fill / ty->size;
}

extern "C" void
rust_intrinsic_ptr_offset(rust_task *task, void **retptr, type_desc *ty,
                          void *ptr, uintptr_t count)
{
    *retptr = &((uint8_t *)ptr)[ty->size * count];
}

extern "C" void
rust_intrinsic_cast(rust_task *task, void *retptr, type_desc *t1,
                    type_desc *t2, void *src)
{
    if (t1->size != t2->size) {
        upcall_fail(task, "attempt to cast values of differing sizes",
                    __FILE__, __LINE__);
        return;
    }

    memmove(retptr, src, t1->size);
}

