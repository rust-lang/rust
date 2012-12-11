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
    fn g(x: uint) -> uint { move x }
}

impl<T> int: A<T> { }

fn f<T, V: A<T>>(i: V, j: uint) -> uint {
    i.g(move j)
}

fn main () {
    assert f::<float, int>(0, 2u) == 2u;
    assert f::<uint, int>(0, 2u) == 2u;
}
