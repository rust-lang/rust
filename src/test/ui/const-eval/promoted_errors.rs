// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(const_err)]

// compile-pass
// compile-flags: -O
fn main() {
    println!("{}", 0u32 - 1);
    let _x = 0u32 - 1;
    //~^ WARN const_err
    println!("{}", 1/(1-1));
    //~^ WARN const_err
    let _x = 1/(1-1);
    //~^ WARN const_err
    //~| WARN const_err
    println!("{}", 1/(false as u32));
    //~^ WARN const_err
    let _x = 1/(false as u32);
    //~^ WARN const_err
    //~| WARN const_err
}
