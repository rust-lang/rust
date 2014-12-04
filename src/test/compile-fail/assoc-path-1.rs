// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we have one and only one associated type per ref.

#![feature(associated_types)]

pub trait Foo {
    type A;
}
pub trait Bar {
    type A;
}

pub fn f1<T>(a: T, x: T::A) {} //~ERROR associated type `A` not found
pub fn f2<T: Foo + Bar>(a: T, x: T::A) {} //~ERROR ambiguous associated type `A`

pub fn main() {}

