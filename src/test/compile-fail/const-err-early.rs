// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_indexing)]
#![deny(const_err)]

pub const A: i8 = -std::i8::MIN; //~ ERROR attempt to negate with overflow
pub const B: u8 = 200u8 + 200u8; //~ ERROR attempt to add with overflow
pub const C: u8 = 200u8 * 4; //~ ERROR attempt to multiply with overflow
pub const D: u8 = 42u8 - (42u8 + 1); //~ ERROR attempt to subtract with overflow
pub const E: u8 = [5u8][1];
//~^ ERROR index out of bounds: the len is 1 but the index is 1

fn main() {
    let _e = [6u8][1];
    //~^ ERROR index out of bounds: the len is 1 but the index is 1
}
