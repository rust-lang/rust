// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#ifndef GPR_H
#define GPR_H

#include "rust_gpr_base.h"

class rust_gpr : public rust_gpr_base {
public:
    uintptr_t r0, r1, r2, r3, r4, r5, r6, r7;
    uintptr_t r8,  r9, r10, r11, r12, r13, r14, r15;
    uintptr_t r16, r17, r18, r19, r20, r21, r22, r23;
    uintptr_t r24, r25, r26, r27, r28, r29, r30, r31;

    inline uintptr_t get_fp() { return r30; }
    inline uintptr_t get_ip() { return r31; }

    inline void set_fp(uintptr_t new_fp) { r30 = new_fp; }
    inline void set_ip(uintptr_t new_ip) { r31 = new_ip; }

    void load();
};

#endif
