// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast
// aux-build:crateresolve_calories-1.rs
// aux-build:crateresolve_calories-2.rs
// error-pattern:mismatched types

// These both have the same version but differ in other metadata
extern mod cr6_1 (name = "crateresolve_calories", vers = "0.1", calories="100");
extern mod cr6_2 (name = "crateresolve_calories", vers = "0.1", calories="200");

pub fn main() {
    assert!(cr6_1::f() == 100);
    assert!(cr6_2::f() == 200);
}
