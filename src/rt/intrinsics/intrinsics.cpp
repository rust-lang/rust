#include "../rust_internal.h"

extern "C" size_t
rust_intrinsic_vec_len(rust_task *task, type_desc *ty, rust_vec *v)
{
    return v->fill / ty->size;
}

