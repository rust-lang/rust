// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-tab

#![warn(unused_mut, unused_parens)] // UI tests pass `-A unused`—see Issue #43896
#![feature(no_debug)]

#[no_mangle] const DISCOVERY: usize = 1;
//~^ ERROR const items should never be #[no_mangle]
//~| HELP try a static value

#[no_mangle]
//~^ HELP remove this attribute
pub fn defiant<T>(_t: T) {}
//~^ WARN functions generic over types must be mangled

#[no_mangle]
fn rio_grande() {}

mod badlands {
    // The private-no-mangle lints shouldn't suggest inserting `pub` when the
    // item is already `pub` (but triggered the lint because, e.g., it's in a
    // private module). (Issue #47383)
    #[no_mangle] pub const DAUNTLESS: bool = true;
    //~^ ERROR const items should never be #[no_mangle]
    //~| HELP try a static value
    #[no_mangle] pub fn val_jean<T>() {}
    //~^ WARN functions generic over types must be mangled
    //~| HELP remove this attribute

    // ... but we can suggest just-`pub` instead of restricted
    #[no_mangle] pub(crate) const VETAR: bool = true;
    //~^ ERROR const items should never be #[no_mangle]
    //~| HELP try a static value
    #[no_mangle] pub(crate) fn crossfield<T>() {}
    //~^ WARN functions generic over types must be mangled
    //~| HELP remove this attribute
}

struct Equinox {
    warp_factor: f32,
}

#[no_debug] // should suggest removal of deprecated attribute
//~^ WARN deprecated
//~| HELP remove this attribute
fn main() {
    while true {
    //~^ WARN denote infinite loops
    //~| HELP use `loop`
        let mut registry_no = (format!("NX-{}", 74205));
        //~^ WARN does not need to be mutable
        //~| HELP remove this `mut`
        //~| WARN unnecessary parentheses
        //~| HELP remove these parentheses
        // the line after `mut` has a `\t` at the beginning, this is on purpose
        let mut
	        b = 1;
        //~^^ WARN does not need to be mutable
        //~| HELP remove this `mut`
        let d = Equinox { warp_factor: 9.975 };
        match d {
            Equinox { warp_factor: warp_factor } => {}
            //~^ WARN this pattern is redundant
            //~| HELP remove this
        }
        println!("{} {}", registry_no, b);
    }
}
