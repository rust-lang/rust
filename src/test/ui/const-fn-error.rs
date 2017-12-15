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
    let mut sum = 0; //~ ERROR blocks in constant functions are limited
    for i in 0..x { //~ ERROR calls in constant functions
    //~| ERROR constant function contains unimplemented
        sum += i;
    }
    sum //~ ERROR E0080
        //~| non-constant path in constant
}

#[allow(unused_variables)]
fn main() {
    let a : [i32; f(X)];
    //~^ WARNING constant evaluation error: non-constant path
}
