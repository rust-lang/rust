// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(impl_trait_in_bindings)]

use std::fmt::Debug;

const FOO: impl Debug + Clone + PartialEq<i32> = 42;

static BAR: impl Debug + Clone + PartialEq<i32> = 42;

fn a<T: Clone>(x: T) {
    let y: impl Clone = x;
    let _ = y.clone();
}

fn b<T: Clone>(x: T) {
    let f = move || {
        let y: impl Clone = x;
        let _ = y.clone();
    };
    f();
}

trait Foo<T: Clone> {
    fn a(x: T) {
        let y: impl Clone = x;
        let _ = y.clone();
    }
}

impl<T: Clone> Foo<T> for i32 {
    fn a(x: T) {
        let y: impl Clone = x;
        let _ = y.clone();
    }
}

fn main() {
    let foo: impl Debug + Clone + PartialEq<i32> = 42;

    assert_eq!(FOO.clone(), 42);
    assert_eq!(BAR.clone(), 42);
    assert_eq!(foo.clone(), 42);

    a(42);
    b(42);
    i32::a(42);
}
