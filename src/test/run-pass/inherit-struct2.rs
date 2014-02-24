// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test struct inheritance on structs from another crate.

// aux-build:inherit_struct_lib.rs
extern crate inherit_struct_lib;

pub fn main() {
    let s = inherit_struct_lib::S2{f1: 115, f2: 113};
    assert!(s.f1 == 115);
    assert!(s.f2 == 113);

    assert!(inherit_struct_lib::glob_s.f1 == 32);
    assert!(inherit_struct_lib::glob_s.f2 == -45);

    inherit_struct_lib::test_s2(s);
}
