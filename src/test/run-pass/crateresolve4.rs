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
// aux-build:crateresolve4a-1.rs
// aux-build:crateresolve4a-2.rs
// aux-build:crateresolve4b-1.rs
// aux-build:crateresolve4b-2.rs

pub mod a {
    extern mod crateresolve4b = "crateresolve4b#0.1";
    pub fn f() { assert!(crateresolve4b::f() == 20); }
}

pub mod b {
    extern mod crateresolve4b = "crateresolve4b#0.2";
    pub fn f() { assert!(crateresolve4b::g() == 10); }
}

pub fn main() {
    a::f();
    b::f();
}
