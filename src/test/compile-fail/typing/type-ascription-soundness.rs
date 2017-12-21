// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Type ascription doesn't lead to unsoundness

#![feature(type_ascription)]

fn main() {
    let arr = &[1u8, 2, 3];
    let ref x = arr: &[u8]; //~ ERROR mismatched types
    let ref mut x = arr: &[u8]; //~ ERROR mismatched types
    match arr: &[u8] { //~ ERROR mismatched types
        ref x => {}
    }
    let _len = (arr: &[u8]).len(); //~ ERROR mismatched types
}
