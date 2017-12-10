// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn closure_to_loc() {
    let mut x = |c| c + 1;
    x = |c| c + 1;
    //~^ ERROR mismatched types
    //~| NOTE no two closures, even if identical, have the same type
    //~| HELP consider boxing your closure and/or using it as a trait object
    //~| expected closure, found a different closure
    //~| expected type `[closure
    //~| found type `[closure
}

fn closure_from_match() {
    let x = match 1usize {
        1 => |c| c + 1,
        2 => |c| c - 1,
        //~^ NOTE match arm with an incompatible type
        _ => |c| c - 1
    };
    //~^^^^^^ ERROR match arms have incompatible types
    //~| NOTE no two closures, even if identical, have the same type
    //~| HELP consider boxing your closure and/or using it as a trait object
    //~| expected closure, found a different closure
    //~| expected type `[closure
    //~| found type `[closure
}

fn main() { }
