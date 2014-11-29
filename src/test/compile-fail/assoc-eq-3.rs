// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test equality constraints on associated types. Check we get type errors
// where we should.

#![feature(associated_types)]

pub trait Foo {
    type A;
    fn boo(&self) -> <Self as Foo>::A;
}

struct Bar;

impl Foo for int {
    type A = uint;
    fn boo(&self) -> uint {
        42
    }
}

fn foo1<I: Foo<A=Bar>>(x: I) {
    let _: Bar = x.boo();
}

fn foo2<I: Foo>(x: I) {
    let _: Bar = x.boo(); //~ERROR mismatched types
}


pub fn baz(x: &Foo<A=Bar>) {
    let _: Bar = x.boo();
}


pub fn main() {
    let a = 42i;
    foo1(a); //~ERROR the trait `Foo` is not implemented for the type `int`
    baz(&a); //~ERROR the trait `Foo` is not implemented for the type `int`
}
