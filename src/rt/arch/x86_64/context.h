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
// register are found in "regs.h".  Note that the alignment must be
// 16 bytes so that SSE instructions can be used.
#include "regs.h"
struct registers_t {
    uint64_t data[RUSTRT_MAX];
} __attribute__((aligned(16)));

extern "C" void __morestack(void *args, void *fn_ptr, uintptr_t stack_ptr);

class context {
public:
    registers_t regs;
    
    context();
    
    context *next;
    
    void swap(context &out);
    void call(void *f, void *arg, void *sp);

    void call_shim_on_c_stack(void *args, void *fn_ptr) {
        __morestack(args, fn_ptr, regs.data[RUSTRT_RSP]);
    }
};

#endif
