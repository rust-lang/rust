// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass
// aux-build:local_inner_macros.rs

#![feature(use_extern_macros)]

extern crate local_inner_macros;

use local_inner_macros::{public_macro, public_macro_dynamic};

public_macro!();

macro_rules! local_helper {
    () => ( struct Z; )
}

public_macro_dynamic!(local_helper);

fn main() {
    let s = S;
    let z = Z;
}
