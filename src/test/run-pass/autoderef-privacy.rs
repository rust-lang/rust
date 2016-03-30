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

#![allow(unused)]

#[derive(Default)]
pub struct Bar2 { i: i32 }
#[derive(Default)]
pub struct Baz2(i32);

impl Bar2 {
    fn f(&self) -> bool { true }
}

mod foo {
    #[derive(Default)]
    pub struct Bar { i: ::Bar2 }
    #[derive(Default)]
    pub struct Baz(::Baz2);

    impl Bar {
        fn f(&self) -> bool { false }
    }

    impl ::std::ops::Deref for Bar {
        type Target = ::Bar2;
        fn deref(&self) -> &::Bar2 { &self.i }
    }

    impl ::std::ops::Deref for Baz {
        type Target = ::Baz2;
        fn deref(&self) -> &::Baz2 { &self.0 }
    }

    pub fn f(bar: &Bar, baz: &Baz) {
        // Since the private fields and methods are visible here, there should be no autoderefs.
        let _: &::Bar2 = &bar.i;
        let _: &::Baz2 = &baz.0;
        assert!(!bar.f());
    }
}

fn main() {
    let bar = foo::Bar::default();
    let baz = foo::Baz::default();
    foo::f(&bar, &baz);

    let _: i32 = bar.i;
    let _: i32 = baz.0;
    assert!(bar.f());
}
