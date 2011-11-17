// Use `clang++ -emit-llvm -S -arch i386 -O3 -I../isaac -I../uthash
//      -I../arch/i386 -fno-stack-protector -o intrinsics.ll intrinsics.cpp`

#include "../rust_internal.h"
#include "../rust_scheduler.h"
#include <cstdlib>
#include <cstring>

extern "C" CDECL void
upcall_fail(char const *expr, char const *file, size_t line);

extern "C" CDECL void
port_recv(uintptr_t *dptr, rust_port *port);

extern "C" CDECL void
rust_task_sleep(size_t time_in_us);

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
    if (t1->size != t2->size) {
        upcall_fail("attempt to cast values of differing sizes",
                    __FILE__, __LINE__);
        return;
    }

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
                          size_t time_in_us) {
    rust_task_sleep(time_in_us);
}

