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

// test conflicting implementation error
// should an intersection impl help be shown?

trait MyClone {
    fn clone(&self) -> Self;
}

impl<T: Copy> MyClone for T {
    fn clone(&self) -> T {
        *self
    }
}

impl<T: Clone> MyClone for Option<T> {
    fn clone(&self) -> Option<T> {
        match *self {
            Some(ref v) => Some(v.clone()),
            None => None,
        }
    }
}
//~^^^^^^^^ ERROR conflicting implementations of trait `MyClone` for type `std::option::Option<_>`

fn main() {}
