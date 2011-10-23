// Use `clang++ -emit-llvm -S -arch i386 -O3 -I../isaac -I../uthash
//      -I../arch/i386 -fno-stack-protector -o intrinsics.ll intrinsics.cpp`

#include "../rust_internal.h"
#include "../rust_scheduler.h"
#include <cstdlib>
#include <cstring>

extern "C" CDECL void
upcall_fail(char const *expr, char const *file, size_t line);

extern "C" void
rust_intrinsic_vec_len(rust_task *task, size_t *retptr, type_desc *ty,
                       rust_vec **vp)
{
    *retptr = (*vp)->fill / ty->size;
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
        upcall_fail("attempt to cast values of differing sizes",
                    __FILE__, __LINE__);
        return;
    }

    memmove(retptr, src, t1->size);
}

extern "C" void
rust_intrinsic_addr_of(rust_task *task, void **retptr, type_desc *ty,
                       void *valptr) {
    *retptr = valptr;
}

extern "C" void
rust_intrinsic_recv(rust_task *task, void **retptr, type_desc *ty,
                    rust_port *port) {
    port_recv((uintptr_t*)retptr, port);
}

extern "C" void
rust_intrinsic_get_type_desc(rust_task *task, void **retptr,
                             type_desc* ty) {
    *(type_desc**)retptr = ty;
}

