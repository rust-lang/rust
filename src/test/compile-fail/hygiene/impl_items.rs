// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty pretty-printing is unhygienic

#![feature(decl_macro)]

mod foo {
    struct S;
    impl S {
        fn f(&self) {}
    }

    pub macro m() {
        let _: () = S.f(); //~ ERROR type `for<'r> fn(&'r foo::S) {foo::S::f}` is private
    }
}

struct S;

macro m($f:ident) {
    impl S {
        fn f(&self) -> u32 { 0 }
        fn $f(&self) -> i32 { 0 }
    }
    fn f() {
        let _: u32 = S.f();
        let _: i32 = S.$f();
    }
}

m!(f);

fn main() {
    let _: i32 = S.f();
    foo::m!();
}
