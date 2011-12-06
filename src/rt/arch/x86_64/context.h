// -*- mode: c++ -*-

#ifndef CONTEXT_H
#define CONTEXT_H

#include <cstdlib>
#include <inttypes.h>
#include <stdint.h>
#include <xmmintrin.h>

#include "vg/memcheck.h"

template<typename T>
T align_down(T sp)
{
    // There is no platform we care about that needs more than a
    // 16-byte alignment.
    return (T)((uint64_t)sp & ~(16 - 1));
}

// The struct in which we store the saved data.  This is mostly the
// volatile registers and instruction pointer, but it also includes
// RCX/RDI which are used to pass arguments.  The indices for each
// register are found in "regs.h":
#include "regs.h"
struct registers_t {
    uint64_t data[RUSTRT_MAX];
};

extern "C" void asm_call_on_stack(void *args, void *fn_ptr, uintptr_t stack_ptr);

class context {
public:
    registers_t regs;
    
    context();
    
    context *next;
    
    void swap(context &out);
    void call(void *f, void *arg, void *sp);
    void call(void *f, void *sp);
    
    // Note that this doesn't actually adjust esp. Instead, we adjust esp when
    // we actually do the call. This is needed for exception safety -- if the
    // function being called causes the task to fail, then we have to avoid
    // leaking space on the C stack.
    inline void *alloc_stack(size_t nbytes) {
        uint64_t bot = regs.data[RUSTRT_RSP];
        uint64_t top = align_down(bot - nbytes);

        (void)VALGRIND_MAKE_MEM_UNDEFINED(top - 4, bot - top + 4);

        return reinterpret_cast<void *>(top);
    }

    void call_shim_on_c_stack(void *args, void *fn_ptr) {
        asm_call_on_stack(args, fn_ptr, regs.data[RUSTRT_RSP]);
    }
};

#endif
