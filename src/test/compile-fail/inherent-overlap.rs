// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that you cannot define items with the same name in overlapping inherent
// impl blocks.

#![allow(unused)]

struct Foo;

impl Foo {
    fn id() {} //~ ERROR duplicate definitions
    //~^ WARN previously accepted
}

impl Foo {
    fn id() {}
}

struct Bar<T>(T);

impl<T> Bar<T> {
    fn bar(&self) {} //~ ERROR duplicate definitions
    //~^ WARN previously accepted
}

impl Bar<u32> {
    fn bar(&self) {}
}

struct Baz<T>(T);

impl<T: Copy> Baz<T> {
    fn baz(&self) {} //~ ERROR duplicate definitions
    //~^ WARN previously accepted
}

impl<T> Baz<Vec<T>> {
    fn baz(&self) {}
}

fn main() {}
