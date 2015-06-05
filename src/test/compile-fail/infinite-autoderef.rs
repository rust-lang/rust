// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

use std::ops::Deref;

struct Foo;

impl Deref for Foo {
    type Target = Foo;

    fn deref(&self) -> &Foo {
        self
    }
}

pub fn main() {
    let mut x;
    loop {
        x = box x; //~ ERROR mismatched types
        x.foo; //~ ERROR the type of this value must be known in this context
        x.bar();
    }

    Foo.foo;
    //~^ ERROR reached the recursion limit while auto-dereferencing Foo
    //~| ERROR reached the recursion limit while auto-dereferencing Foo
    //~| ERROR attempted access of field `foo` on type `Foo`, but no field with that name was
    // found
    Foo.bar();
    //~^ ERROR reached the recursion limit while auto-dereferencing Foo
    //~| ERROR no method named `bar` found for type `Foo` in the current scope
}
