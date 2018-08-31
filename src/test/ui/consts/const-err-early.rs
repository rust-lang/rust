// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(const_err)]

pub const A: i8 = -std::i8::MIN; //~ ERROR const_err
pub const B: u8 = 200u8 + 200u8; //~ ERROR const_err
pub const C: u8 = 200u8 * 4; //~ ERROR const_err
pub const D: u8 = 42u8 - (42u8 + 1); //~ ERROR const_err
pub const E: u8 = [5u8][1]; //~ ERROR const_err

fn main() {
    let _a = A; //~ ERROR erroneous constant used
    let _b = B; //~ ERROR erroneous constant used
    let _c = C; //~ ERROR erroneous constant used
    let _d = D; //~ ERROR erroneous constant used
    let _e = E; //~ ERROR erroneous constant used
    let _e = [6u8][1]; //~ ERROR index out of bounds
}
