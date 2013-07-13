// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


trait A<T> {
    fn g(&self, x: uint) -> uint { x }
}

impl<T> A<T> for int { }

fn f<T, V: A<T>>(i: V, j: uint) -> uint {
    i.g(j)
}

pub fn main () {
    assert_eq!(f::<float, int>(0, 2u), 2u);
    assert_eq!(f::<uint, int>(0, 2u), 2u);
}
