// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check we do not select a private method or field when computing autoderefs

#![feature(rustc_attrs)]
#![allow(unused)]

pub struct Bar2 { i: i32 }
pub struct Baz2(i32);

impl Bar2 {
    fn f(&self) {}
}

mod foo {
    pub struct Bar { i: i32 }
    pub struct Baz(i32);

    impl Bar {
        fn f(&self) {}
    }

    impl ::std::ops::Deref for Bar {
        type Target = ::Bar2;
        fn deref(&self) -> &::Bar2 { unimplemented!() }
    }

    impl ::std::ops::Deref for Baz {
        type Target = ::Baz2;
        fn deref(&self) -> &::Baz2 { unimplemented!() }
    }
}

fn f(bar: foo::Bar, baz: foo::Baz) {
    let _ = bar.i;
    let _ = baz.0;
    let _ = bar.f();
}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
