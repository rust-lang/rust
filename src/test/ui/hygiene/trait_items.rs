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

mod foo {
    pub trait T {
        fn f(&self) {}
    }
    impl T for () {}
}

mod bar {
    use foo::*;
    pub macro m() { ().f() }
    fn f() { ::baz::m!(); }
}

mod baz {
    pub macro m() { ().f() } //~ ERROR no method named `f` found for type `()` in the current scope
    fn f() { ::bar::m!(); }
}

fn main() {}
