// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Equal {
    fn equal(&self, other: &Self) -> bool;
    fn equals<T,U>(&self, this: &T, that: &T, x: &U, y: &U) -> bool
            where T: Eq, U: Eq;
}

impl<T> Equal for T where T: Eq {
    fn equal(&self, other: &T) -> bool {
        self == other
    }
    fn equals<U,X>(&self, this: &U, other: &U, x: &X, y: &X) -> bool
            where U: Eq, X: Eq {
        this == other && x == y
    }
}

pub fn equal<T>(x: &T, y: &T) -> bool where T: Eq {
    x == y
}
