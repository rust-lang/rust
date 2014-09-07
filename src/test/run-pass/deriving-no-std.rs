// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![no_std]

extern crate core;
extern crate rand;
extern crate native;
extern crate serialize;
extern crate collections;

// Issue #16803

#[deriving(Clone, Hash, Encodable, Decodable, PartialEq, Eq, PartialOrd, Ord,
           Rand, Show, Zero, Default)]
struct Foo {
    x: uint,
}

// needed for Zero
impl core::ops::Add<Foo, Foo> for Foo {
    fn add(&self, rhs: &Foo) -> Foo {
        Foo { x: self.x + rhs.x }
    }
}

fn main() {
    Foo { x: 0 };
}
