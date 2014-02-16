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
// aux-build:crateresolve2-1.rs
// aux-build:crateresolve2-2.rs
// aux-build:crateresolve2-3.rs

mod a {
    extern crate crateresolve2 = "crateresolve2#0.1";
    pub fn f() { assert!(crateresolve2::f() == 10); }
}

mod b {
    extern crate crateresolve2 = "crateresolve2#0.2";
    pub fn f() { assert!(crateresolve2::f() == 20); }
}

mod c {
    extern crate crateresolve2 = "crateresolve2#0.3";
    pub fn f() { assert!(crateresolve2::f() == 30); }
}

pub fn main() {
    a::f();
    b::f();
    c::f();
}
