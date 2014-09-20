// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Testing that supertrait methods can be called on subtrait object types

trait Foo {
    fn f(&self) -> int;
}

trait Bar : Foo {
    fn g(&self) -> int;
}

struct A {
    x: int
}

impl Foo for A {
    fn f(&self) -> int { 10 }
}

impl Bar for A {
    fn g(&self) -> int { 20 }
}

pub fn main() {
    let a = &A { x: 3 };
    let afoo = a as &Foo;
    let abar = a as &Bar;
    assert_eq!(afoo.f(), 10);
    assert_eq!(abar.g(), 20);
    assert_eq!(abar.f(), 10);
}
