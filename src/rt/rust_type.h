
#ifndef RUST_TYPE_H
#define RUST_TYPE_H

#include "rust_refcount.h"

// The type of functions that we spawn, which fall into two categories:
// - the main function: has a NULL environment, but uses the void* arg
// - unique closures of type fn~(): have a non-NULL environment, but
//   no arguments (and hence the final void*) is harmless
typedef void (*CDECL spawn_fn)(void*, rust_opaque_box*, void *);

struct type_desc;

typedef void CDECL (glue_fn)(void *, void *, const type_desc **, void *);

struct rust_shape_tables {
    uint8_t *tags;
    uint8_t *resources;
};

// Corresponds to the boxed data in the @ region.  The body follows the
// header; you can obtain a ptr via box_body() below.
struct rust_opaque_box {
    ref_cnt_t ref_count;
    type_desc *td;
    rust_opaque_box *prev;
    rust_opaque_box *next;
};

// corresponds to the layout of a fn(), fn@(), fn~() etc
struct fn_env_pair {
    spawn_fn f;
    rust_opaque_box *env;
};

static inline void *box_body(rust_opaque_box *box) {
    // Here we take advantage of the fact that the size of a box in 32
    // (resp. 64) bit is 16 (resp. 32) bytes, and thus always 16-byte aligned.
    // If this were to change, we would have to update the method
    // rustc::middle::trans::base::opaque_box_body() as well.
    return (void*)(box + 1);
}

struct type_desc {
    // First part of type_desc is known to compiler.
    // first_param = &descs[1] if dynamic, null if static.
    const type_desc **first_param;
    size_t size;
    size_t align;
    glue_fn *take_glue;
    glue_fn *drop_glue;
    glue_fn *free_glue;
    void *UNUSED;
    glue_fn *sever_glue;    // For GC.
    glue_fn *mark_glue;     // For GC.
    uintptr_t unused2;
    void *UNUSED_2;
    const uint8_t *shape;
    const rust_shape_tables *shape_tables;
    uintptr_t n_params;
    uintptr_t n_obj_params;

    // Residual fields past here are known only to runtime.
    UT_hash_handle hh;
    size_t n_descs;
    const type_desc *descs[];
};

extern "C" type_desc *rust_clone_type_desc(type_desc*);

#endif
