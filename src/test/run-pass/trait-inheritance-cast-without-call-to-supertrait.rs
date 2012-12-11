// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Testing that we can cast to a subtrait and call subtrait
// methods. Not testing supertrait methods

trait Foo {
    fn f() -> int;
}

trait Bar : Foo {
    fn g() -> int;
}

struct A {
    x: int
}

impl A : Foo {
    fn f() -> int { 10 }
}

impl A : Bar {
    fn g() -> int { 20 }
}

fn main() {
    let a = &A { x: 3 };
    let afoo = a as &Foo;
    let abar = a as &Bar;
    assert afoo.f() == 10;
    assert abar.g() == 20;
}

