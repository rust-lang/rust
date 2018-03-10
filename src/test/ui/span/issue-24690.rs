// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A test to ensure that helpful `note` messages aren't emitted more often
//! than necessary.

#![feature(rustc_attrs)]

// Although there are three warnings, we should only get two "lint level defined
// here" notes pointing at the `warnings` span, one for each error type.
#![warn(unused)]

#[rustc_error]
fn main() { //~ ERROR compilation successful
    let theTwo = 2; //~ WARN should have a snake case name
    let theOtherTwo = 2; //~ WARN should have a snake case name
    //~^ WARN unused variable
    println!("{}", theTwo);
}
