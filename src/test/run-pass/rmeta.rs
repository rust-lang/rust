// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that using rlibs and rmeta dep crates work together. Specifically, that
// there can be both an rmeta and an rlib file and rustc will prefer the rlib.

// aux-build:rmeta_rmeta.rs
// aux-build:rmeta_rlib.rs

extern crate rmeta_aux;
use rmeta_aux::Foo;

pub fn main() {
    let _ = Foo { field: 42 };
}
