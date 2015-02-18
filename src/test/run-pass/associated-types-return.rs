// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test equality constraints on associated types in a where clause.

pub trait Foo {
    type A;
    fn boo(&self) -> <Self as Foo>::A;
}

#[derive(PartialEq)]
struct Bar;

impl Foo for int {
    type A = uint;
    fn boo(&self) -> uint { 42 }
}

impl Foo for Bar {
    type A = int;
    fn boo(&self) -> int { 43 }
}

impl Foo for char {
    type A = Bar;
    fn boo(&self) -> Bar { Bar }
}

fn foo1<I: Foo<A=Bar>>(x: I) -> Bar {
    x.boo()
}

fn foo2<I: Foo>(x: I) -> <I as Foo>::A {
    x.boo()
}

pub fn main() {
    let a = 42;
    assert!(foo2(a) == 42_usize);

    let a = Bar;
    assert!(foo2(a) == 43);

    let a = 'a';
    foo1(a);
    assert!(foo2(a) == Bar);
}
