// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// test that certain things are disallowed in constant functions

#![feature(const_fn)]

// no destructuring
const fn i((
            a,
            //~^ ERROR arguments of constant functions can only be immutable by-value bindings
            b
            //~^ ERROR arguments of constant functions can only be immutable by-value bindings
           ): (u32, u32)) -> u32 {
    a + b
    //~^ ERROR let bindings in constant functions are unstable
    //~| ERROR let bindings in constant functions are unstable
}

fn main() {}
