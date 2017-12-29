// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can't call random fns in a const fn or do other bad things.

#![feature(const_fn)]

use std::mem::transmute;

fn random() -> u32 { 0 }

const fn sub(x: &u32) -> usize {
    unsafe { transmute(x) } //~ ERROR E0015
}

const fn sub1() -> u32 {
    random() //~ ERROR E0015
}

static Y: u32 = 0;

const fn get_Y() -> u32 {
    Y
        //~^ ERROR E0013
        //~| ERROR cannot refer to statics by value
}

const fn get_Y_addr() -> &'static u32 {
    &Y
        //~^ ERROR E0013
}

const fn get() -> u32 {
    let x = 22; //~ ERROR E0016
    let y = 44; //~ ERROR E0016
    x + y
}

fn main() {
}
