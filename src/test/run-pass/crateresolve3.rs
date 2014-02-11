// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast
// aux-build:crateresolve3-1.rs
// aux-build:crateresolve3-2.rs

// verify able to link with crates with same name but different versions
// as long as no name collision on invoked functions.

mod a {
    extern mod crateresolve3 = "crateresolve3#0.1";
    pub fn f() { assert!(crateresolve3::f() == 10); }
}

mod b {
    extern mod crateresolve3 = "crateresolve3#0.2";
    pub fn f() { assert!(crateresolve3::g() == 20); }
}

pub fn main() {
    a::f();
    b::f();
}
