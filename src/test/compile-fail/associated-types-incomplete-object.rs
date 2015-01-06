// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that the user gets an errror if they omit a binding from an
// object type.

pub trait Foo {
    type A;
    type B;
    fn boo(&self) -> <Self as Foo>::A;
}

struct Bar;

impl Foo for int {
    type A = uint;
    type B = char;
    fn boo(&self) -> uint {
        42
    }
}

pub fn main() {
    let a = &42i as &Foo<A=uint, B=char>;

    let b = &42i as &Foo<A=uint>;
    //~^ ERROR the value of the associated type `B` (from the trait `Foo`) must be specified

    let c = &42i as &Foo<B=char>;
    //~^ ERROR the value of the associated type `A` (from the trait `Foo`) must be specified

    let d = &42i as &Foo;
    //~^ ERROR the value of the associated type `A` (from the trait `Foo`) must be specified
    //~| ERROR the value of the associated type `B` (from the trait `Foo`) must be specified
}
