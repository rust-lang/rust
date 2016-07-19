// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Add;

trait BrokenAdd: Copy + Add<Output=Self> {
    fn broken_add<T>(&self, rhs: T) -> Self {
        *self + rhs //~  ERROR mismatched types
                    //~| expected type `Self`
                    //~| found type `T`
                    //~| expected Self, found type parameter
    }
}

impl<T: Copy + Add<Output=T>> BrokenAdd for T {}

pub fn main() {
    let foo: u8 = 0;
    let x: u8 = foo.broken_add("hello darkness my old friend".to_string());
    println!("{}", x);
}
