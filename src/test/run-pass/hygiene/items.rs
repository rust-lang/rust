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

pub macro m($foo:ident, $f:ident, $e:expr) {
    mod foo {
        pub fn f() -> u32 { 0 }
        pub fn $f() -> u64 { 0 }
    }

    mod $foo {
        pub fn f() -> i32 { 0 }
        pub fn $f() -> i64 { 0  }
    }

    let _: u32 = foo::f();
    let _: u64 = foo::$f();
    let _: i32 = $foo::f();
    let _: i64 = $foo::$f();
    let _: i64 = $e;
}

fn main() {
    m!(foo, f, foo::f());
    let _: i64 = foo::f();
}
