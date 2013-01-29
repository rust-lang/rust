// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#define RUSTRT_MAX  32

// ARG0 is the register in which the first argument goes.
// Naturally this depends on your operating system.
#define RUSTRT_ARG0_S r4
#define RUSTRT_ARG1_S r5
#define RUSTRT_ARG2_S r6
#define RUSTRT_ARG3_S r7
