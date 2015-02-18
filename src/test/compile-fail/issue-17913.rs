// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// error-pattern: too big for the current architecture

#![feature(box_syntax)]

#[cfg(target_pointer_width = "64")]
fn main() {
    let n = 0_usize;
    let a = box [&n; 0xF000000000000000_usize];
    println!("{}", a[0xFFFFFF_usize]);
}

#[cfg(target_pointer_width = "32")]
fn main() {
    let n = 0_usize;
    let a = box [&n; 0xFFFFFFFF_usize];
    println!("{}", a[0xFFFFFF_usize]);
}
