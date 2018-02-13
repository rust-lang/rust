// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(unused_mut, unused_parens)] // UI tests pass `-A unused`—see Issue #43896
#![feature(no_debug)]

#[no_mangle] static SHENZHOU: usize = 1; // should suggest `pub`
//~^ WARN static is marked #[no_mangle]
#[no_mangle] const DISCOVERY: usize = 1; // should suggest `pub static` rather than `const`
//~^ ERROR const items should never be #[no_mangle]

#[no_mangle] // should suggest removal (generics can't be no-mangle)
pub fn defiant<T>(_t: T) {}
//~^ WARN functions generic over types must be mangled

#[no_mangle]
fn rio_grande() {} // should suggest `pub`
//~^ WARN function is marked

mod badlands {
    // The private-no-mangle lints shouldn't suggest inserting `pub` when the
    // item is already `pub` (but triggered the lint because, e.g., it's in a
    // private module). (Issue #47383)
    #[no_mangle] pub static DAUNTLESS: bool = true;
    //~^ WARN static is marked
    #[no_mangle] pub fn val_jean() {}
    //~^ WARN function is marked
}

struct Equinox {
    warp_factor: f32,
}

#[no_debug] // should suggest removal of deprecated attribute
//~^ WARN deprecated
fn main() {
    while true { // should suggest `loop`
    //~^ WARN denote infinite loops
        let mut a = (1); // should suggest no `mut`, no parens
        //~^ WARN does not need to be mutable
        //~| WARN unnecessary parentheses
        let d = Equinox { warp_factor: 9.975 };
        match d {
            Equinox { warp_factor: warp_factor } => {} // should suggest shorthand
            //~^ WARN this pattern is redundant
        }
        println!("{}", a);
    }
}
