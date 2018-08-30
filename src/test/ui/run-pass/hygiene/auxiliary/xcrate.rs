// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(decl_macro)]
#![allow(unused)]

pub use bar::test;

extern crate std as foo;

pub fn f() {}
use f as f2;

mod bar {
    pub fn g() {}
    use baz::h;

    pub macro test() {
        use std::mem;
        use foo::cell;
        ::f();
        ::f2();
        g();
        h();
        ::bar::h();
    }
}

mod baz {
    pub fn h() {}
}
