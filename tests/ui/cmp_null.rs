// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::cmp_null)]
#![allow(unused_mut)]

use std::ptr;

fn main() {
    let x = 0;
    let p: *const usize = &x;
    if p == ptr::null() {
        println!("This is surprising!");
    }
    let mut y = 0;
    let mut m: *mut usize = &mut y;
    if m == ptr::null_mut() {
        println!("This is surprising, too!");
    }
}
