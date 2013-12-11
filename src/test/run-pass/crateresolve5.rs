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
// aux-build:crateresolve5-1.rs
// aux-build:crateresolve5-2.rs

extern mod cr5_1 = "crateresolve5#0.1";
extern mod cr5_2 = "crateresolve5#0.2";

pub fn main() {
    // Structural types can be used between two versions of the same crate
    assert!(cr5_1::struct_nameval().name == cr5_2::struct_nameval().name);
    assert!(cr5_1::struct_nameval().val == cr5_2::struct_nameval().val);
    // Make sure these are actually two different crates
    assert!(cr5_1::f() == 10 && cr5_2::f() == 20);
}
