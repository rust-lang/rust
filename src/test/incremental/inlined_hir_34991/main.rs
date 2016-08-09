// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #34991: an ICE occurred here because we inline
// some of the vector routines and give them a local def-id `X`. This
// got hashed after trans (`Hir(X)`). When we load back up, we get an
// error because the `X` is remapped to the original def-id (in
// libstd), and we can't hash a HIR node from std.

// revisions:rpass1 rpass2

#![feature(rustc_attrs)]

use std::vec::Vec;

pub fn foo() -> Vec<i32> {
    vec![1, 2, 3]
}

pub fn bar() {
    foo();
}

pub fn main() {
    bar();
}
