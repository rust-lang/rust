// Rust intrinsics. These are built into each compilation unit and are
// run on the Rust stack. They should not call C methods because that
// will very likely result in running off the end of the stack.
// Build with the script in src/etc/gen-intrinsics

#include "../rust_internal.h"
#include "../rust_scheduler.h"
#include <cstdlib>
#include <cstring>

extern "C" CDECL void
port_recv(uintptr_t *dptr, rust_port *port);

extern "C" CDECL void
rust_task_sleep(rust_task *task, size_t time_in_us);

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

extern "C" void
rust_intrinsic_recv(void **retptr,
                    void *env,
                    type_desc *ty,
                    rust_port *port) {
    port_recv((uintptr_t*)retptr, port);
}

extern "C" void
rust_intrinsic_get_type_desc(void **retptr,
                             void *env,
                             type_desc* ty) {
    *(type_desc**)retptr = ty;
}

extern "C" void
rust_intrinsic_task_sleep(void **retptr,
                          void *env,
			  rust_task *task,
                          size_t time_in_us) {
    rust_task_sleep(task, time_in_us);
}

