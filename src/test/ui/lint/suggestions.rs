// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(unused_mut)] // UI tests pass `-A unused`—see Issue #43896
#![feature(no_debug)]

#[no_debug] // should suggest removal of deprecated attribute
fn main() {
    while true { // should suggest `loop`
        let mut a = (1); // should suggest no `mut`, no parens
        println!("{}", a);
    }
}
