// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(fn_traits, unboxed_closures)]

use std::ops::FnMut;

struct S {
    x: isize,
    y: isize,
}

impl FnMut<(isize,)> for S {
    extern "rust-call" fn call_mut(&mut self, (z,): (isize,)) -> isize {
        self.x * self.y * z
    }
}

impl FnOnce<(isize,)> for S {
    type Output = isize;
    extern "rust-call" fn call_once(mut self, (z,): (isize,)) -> isize {
        self.call_mut((z,))
    }
}

fn main() {
    let mut s = S {
        x: 3,
        y: 3,
    };
    let ans = s("what");    //~ ERROR mismatched types
    //~^ NOTE expected isize, found reference
    //~| NOTE expected type
    //~| NOTE found type
    let ans = s();
    //~^ ERROR this function takes 1 parameter but 0 parameters were supplied
    //~| NOTE expected 1 parameter
    let ans = s("burma", "shave");
    //~^ ERROR this function takes 1 parameter but 2 parameters were supplied
    //~| NOTE expected 1 parameter
}
