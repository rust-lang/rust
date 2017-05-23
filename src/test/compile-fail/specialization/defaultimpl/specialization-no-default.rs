// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(specialization)]

// Check a number of scenarios in which one impl tries to override another,
// without correctly using `default`.

////////////////////////////////////////////////////////////////////////////////
// Test 1: one layer of specialization, multiple methods, missing `default`
////////////////////////////////////////////////////////////////////////////////

trait Foo {
    fn foo(&self);
    fn bar(&self);
}

impl<T> Foo for T {
    fn foo(&self) {}
    fn bar(&self) {}
}

impl Foo for u8 {}
impl Foo for u16 {
    fn foo(&self) {} //~ ERROR E0520
}
impl Foo for u32 {
    fn bar(&self) {} //~ ERROR E0520
}

////////////////////////////////////////////////////////////////////////////////
// Test 2: one layer of specialization, missing `default` on associated type
////////////////////////////////////////////////////////////////////////////////

trait Bar {
    type T;
}

impl<T> Bar for T {
    type T = u8;
}

impl Bar for u8 {
    type T = (); //~ ERROR E0520
}

////////////////////////////////////////////////////////////////////////////////
// Test 3a: multiple layers of specialization, missing interior `default`
////////////////////////////////////////////////////////////////////////////////

trait Baz {
    fn baz(&self);
}

default impl<T> Baz for T {
    fn baz(&self) {}
}

impl<T: Clone> Baz for T {
    fn baz(&self) {}
}

impl Baz for i32 {
    fn baz(&self) {} //~ ERROR E0520
}

////////////////////////////////////////////////////////////////////////////////
// Test 3b: multiple layers of specialization, missing interior `default`,
// redundant `default` in bottom layer.
////////////////////////////////////////////////////////////////////////////////

trait Redundant {
    fn redundant(&self);
}

default impl<T> Redundant for T {
    fn redundant(&self) {}
}

impl<T: Clone> Redundant for T {
    fn redundant(&self) {}
}

default impl Redundant for i32 {
    fn redundant(&self) {} //~ ERROR E0520
}

fn main() {}
