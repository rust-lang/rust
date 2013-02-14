// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#ifndef RUST_RNG_H
#define RUST_RNG_H

#include "rand.h"

class rust_kernel;
struct rust_vec_box;

// Initialization helpers for ISAAC RNG

void isaac_seed(rust_kernel* kernel, uint8_t* dest, size_t size);
void isaac_init(rust_kernel *kernel, randctx *rctx, rust_vec_box* user_seed);

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//

#endif
