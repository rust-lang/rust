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

extern "C" void
rust_intrinsic_memmove(void *retptr,
                    void *env,
                    type_desc *ty,
                    void *dst,
                    void *src,
                    uintptr_t count)
{
    memmove(dst, src, ty->size * count);
}

extern "C" void
rust_intrinsic_memcpy(void *retptr,
                    void *env,
                    type_desc *ty,
                    void *dst,
                    void *src,
                    uintptr_t count)
{
    memcpy(dst, src, ty->size * count);
}

extern "C" void
rust_intrinsic_leak(void *retptr,
                    void *env,
                    type_desc *ty,
                    void *thing)
{
}

extern "C" CDECL void *
upcall_shared_realloc(void *ptr, size_t size);

inline void reserve_vec_fast(rust_vec **vpp, size_t size) {
    if (size > (*vpp)->alloc) {
      size_t new_size = next_power_of_two(size);
        size_t alloc_size = new_size + sizeof(rust_vec);
        // Because this is called from an intrinsic we need to use
        // the exported API
        *vpp = (rust_vec*)upcall_shared_realloc(*vpp, alloc_size);
        (*vpp)->alloc = new_size;
    }
}

// Copy elements from one vector to another,
// dealing with reference counts
static inline void
copy_elements(type_desc *elem_t,
              void *pdst, void *psrc, size_t n) {
    char *dst = (char *)pdst, *src = (char *)psrc;
    memmove(dst, src, n);

    // increment the refcount of each element of the vector
    if (elem_t->take_glue) {
        glue_fn *take_glue = elem_t->take_glue;
        size_t elem_size = elem_t->size;
        const type_desc **tydescs = elem_t->first_param;
        for (char *p = dst; p < dst+n; p += elem_size) {
            take_glue(NULL, NULL, tydescs, p);
        }
    }
}

// Because this is used so often, and it calls take glue that must run
// on the rust stack, it is statically compiled into every crate.
extern "C" CDECL void
upcall_intrinsic_vec_push(rust_vec** vp,
			  type_desc* elt_ty, void* elt) {

    size_t new_sz = (*vp)->fill + elt_ty->size;
    reserve_vec_fast(vp, new_sz);
    rust_vec* v = *vp;
    copy_elements(elt_ty, &v->data[0] + v->fill,
                  elt, elt_ty->size);
    v->fill += elt_ty->size;
}

// FIXME: Transational. Remove
extern "C" CDECL void
upcall_vec_push(rust_vec** vp,
		type_desc* elt_ty, void* elt) {
  upcall_intrinsic_vec_push(vp, elt_ty, elt);
}

extern "C" CDECL void
rust_intrinsic_frame_address(void **p, unsigned n) {
    *p = __builtin_frame_address(n);
}

