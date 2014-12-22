// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test equality constraints on associated types. Check we get an error when an
// equality constraint is used in a qualified path.

#![feature(associated_types)]

pub trait Foo {
    type A;
    fn boo(&self) -> <Self as Foo>::A;
}

struct Bar;

impl Foo for int {
    type A = uint;
    fn boo(&self) -> uint { 42 }
}

fn baz<I: Foo>(x: &<I as Foo<A=Bar>>::A) {} //~ERROR equality constraints are not allowed in this

pub fn main() {}
