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

#define LOAD(n) do { \
    uintptr_t tmp; \
    asm(".set noat; move %0, $" #n : "=r" (tmp) :); \
    this->r##n = tmp; \
} while (0)

void rust_gpr::load() {
              LOAD(1); LOAD(2); LOAD(3);
    LOAD(4); LOAD(5); LOAD(6); LOAD(7);

    LOAD(8); LOAD(9); LOAD(10); LOAD(11);
    LOAD(12); LOAD(13); LOAD(14); LOAD(15);

    LOAD(16); LOAD(17); LOAD(18); LOAD(19);
    LOAD(20); LOAD(21); LOAD(22); LOAD(23);

    LOAD(24); LOAD(25); LOAD(26); LOAD(27);
    LOAD(28); LOAD(29); LOAD(30); LOAD(31);
}
