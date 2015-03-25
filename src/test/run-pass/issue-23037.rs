// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core)]

use std::simd::i32x4;
fn main() {
    let foo = i32x4(1,2,3,4);
    let bar = i32x4(40,30,20,10);
    let baz = foo + bar;
    assert!(baz.0 == 41 && baz.1 == 32 && baz.2 == 23 && baz.3 == 14);
}
