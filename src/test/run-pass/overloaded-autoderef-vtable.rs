// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Deref;

struct DerefWithHelper<H, T> {
    helper: H
}

trait Helper<T> {
    fn helper_borrow<'a>(&'a self) -> &'a T;
}

impl<T> Helper<T> for Option<T> {
    fn helper_borrow<'a>(&'a self) -> &'a T {
        self.as_ref().unwrap()
    }
}

impl<T, H: Helper<T>> Deref<T> for DerefWithHelper<H, T> {
    fn deref<'a>(&'a self) -> &'a T {
        self.helper.helper_borrow()
    }
}

struct Foo {x: int}

impl Foo {
    fn foo(&self) -> int {self.x}
}

pub fn main() {
    let x: DerefWithHelper<Option<Foo>, Foo> = DerefWithHelper { helper: Some(Foo {x: 5}) };
    assert!(x.foo() == 5);
}
