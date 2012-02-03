// Rust intrinsics. These are built into each compilation unit and are
// run on the Rust stack. They should not call C methods because that
// will very likely result in running off the end of the stack.
// Build with the script in src/etc/gen-intrinsics

#include "../rust_internal.h"
#include "../rust_util.h"
#include <cstdlib>
#include <cstring>

extern "C" CDECL void
rust_task_yield(rust_task *task, bool *killed);

extern "C" void
rust_intrinsic_vec_len(size_t *retptr,
                         void *env,
                         type_desc *ty,
                         rust_vec **vp)
{
    *retptr = (*vp)->fill / ty->size;
}

extern "C" void
rust_intrinsic_ptr_offset(void **retptr,
                          void *env,
                          type_desc *ty,
                          void *ptr,
                          uintptr_t count)
{
    *retptr = &((uint8_t *)ptr)[ty->size * count];
}

extern "C" void
rust_intrinsic_cast(void *retptr,
                    void *env,
                    type_desc *t1,
                    type_desc *t2,
                    void *src)
{
    // assert t1->size == t2->size
    // FIXME: This should be easily expressible in rust
    memmove(retptr, src, t1->size);
}

extern "C" void
rust_intrinsic_addr_of(void **retptr,
                       void *env,
                       type_desc *ty,
                       void *valptr) {
    *retptr = valptr;
}

struct rust_fn {
    uintptr_t *fn;
    rust_box *env;
};

typedef void (*retptr_fn)(void **retptr,
			   void *env,
			   void **dptr);
// FIXME (1185): This exists just to get access to the return pointer
extern "C" void
rust_intrinsic_call_with_retptr(void **retptr,
				void *env,
				type_desc *ty,
				rust_fn *recvfn) {
    retptr_fn fn = ((retptr_fn)(recvfn->fn));
    ((retptr_fn)(*fn))(NULL, recvfn->env, retptr);
}

extern "C" void
rust_intrinsic_get_type_desc(void **retptr,
                             void *env,
                             type_desc* ty) {
    *(type_desc**)retptr = ty;
}

extern "C" void
rust_intrinsic_task_yield(void **retptr,
                          void *env,
			  rust_task *task,
			  bool *killed) {
    rust_task_yield(task, killed);
}

