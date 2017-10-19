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

#[no_mangle] static SHENZHOU: usize = 1; // should suggest `pub`
#[no_mangle] const DISCOVERY: usize = 1; // should suggest `pub static` rather than `const`

#[no_mangle] // should suggest removal (generics can't be no-mangle)
pub fn defiant<T>(_t: T) {}

#[no_mangle]
fn rio_grande() {} // should suggest `pub`

struct Equinox {
    warp_factor: f32,
}

#[no_debug] // should suggest removal of deprecated attribute
fn main() {
    while true { // should suggest `loop`
        let mut a = (1); // should suggest no `mut`, no parens
        let d = Equinox { warp_factor: 9.975 };
        match d {
            Equinox { warp_factor: warp_factor } => {} // should suggest shorthand
        }
        println!("{}", a);
    }
}
