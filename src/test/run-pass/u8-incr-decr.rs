// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




// -*- rust -*-

// These constants were chosen because they aren't used anywhere
// in the rest of the generated code so they're easily grep-able.
pub fn main() {
    let mut x: u8 = 19u8; // 0x13

    let mut y: u8 = 35u8; // 0x23

    x = x + 7u8; // 0x7

    y = y - 9u8; // 0x9

    assert_eq!(x, y);
}
