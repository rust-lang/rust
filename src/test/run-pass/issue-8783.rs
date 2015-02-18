// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::default::Default;

struct X { pub x: uint }
impl Default for X {
    fn default() -> X {
        X { x: 42_usize }
    }
}

struct Y<T> { pub y: T }
impl<T: Default> Default for Y<T> {
    fn default() -> Y<T> {
        Y { y: Default::default() }
    }
}

fn main() {
    let X { x: _ } = Default::default();
    let Y { y: X { x } } = Default::default();
}
