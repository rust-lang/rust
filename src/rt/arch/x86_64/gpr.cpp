// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "gpr.h"

#define LOAD(rn) do { \
    uintptr_t tmp; \
    asm("movq %%" #rn ",%0" : "=r" (tmp) :); \
    this->rn = tmp; \
} while (0)

void rust_gpr::load() {
    LOAD(rax); LOAD(rbx); LOAD(rcx); LOAD(rdx);
    LOAD(rsi); LOAD(rdi); LOAD(rbp); LOAD(rsi);
    LOAD(r8);  LOAD(r9);  LOAD(r10); LOAD(r11);
    LOAD(r12); LOAD(r13); LOAD(r14); LOAD(r15);
}
