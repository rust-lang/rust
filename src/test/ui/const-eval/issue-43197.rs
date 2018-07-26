// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(const_err)]

#![feature(const_fn)]

const fn foo(x: u32) -> u32 {
    x
}

fn main() {
    const X: u32 = 0-1;
    //~^ WARN this constant cannot be used
    const Y: u32 = foo(0-1);
    //~^ WARN this constant cannot be used
    println!("{} {}", X, Y);
    //~^ ERROR erroneous constant used
    //~| ERROR erroneous constant used
    //~| ERROR E0080
    //~| ERROR E0080
}
