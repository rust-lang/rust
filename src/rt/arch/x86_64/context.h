// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


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

class context {
public:
    registers_t regs;

    context();

    context *next;

    void swap(context &out);
    void call(void *f, void *arg, void *sp);
};

#endif
