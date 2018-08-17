// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #27868

#![feature(nll)]

use std::ops::AddAssign;

struct MyVec<T>(Vec<T>);

impl <T> Drop for MyVec<T> {
    fn drop(&mut self) {
        println!("Being dropped.");
    }
}

impl<T> AddAssign<T> for MyVec<T> {
    fn add_assign(&mut self, _elem: T) {
        println!("In add_assign.");
    }
}

fn main() {
    let mut vec = MyVec(vec![0]);
    let mut vecvec = vec![vec];

    vecvec[0] += {
        vecvec = vec![];
        //~^ ERROR cannot assign to `vecvec` because it is borrowed [E0506]
        0
    };
}
