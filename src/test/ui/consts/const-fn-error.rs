// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_fn)]

const X : usize = 2;

const fn f(x: usize) -> usize {
    let mut sum = 0;
    //~^ let bindings in constant functions are unstable
    //~| statements in constant functions are unstable
    for i in 0..x {
        //~^ ERROR E0015
        //~| ERROR E0019
        sum += i;
    }
    sum
}

#[allow(unused_variables)]
fn main() {
    let a : [i32; f(X)]; //~ ERROR E0080
}
