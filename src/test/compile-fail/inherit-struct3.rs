// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test struct inheritance.
#![feature(struct_inherit)]

virtual struct S1 {
    f1: int,
}

struct S6 : S1 {
    f2: int,
}

pub fn main() {
    let s = S6{f2: 3}; //~ ERROR missing field: `f1`
    let s = S6{f1: 3}; //~ ERROR missing field: `f2`
}
