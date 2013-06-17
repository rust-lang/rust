// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cast;

#[packed]
struct S4(u8,[u8, .. 3]);

#[packed]
struct S5(u8,u32);

fn main() {
    unsafe {
        let s4 = S4(1, [2,3,4]);
        let transd : [u8, .. 4] = cast::transmute(s4);
        assert_eq!(transd, [1, 2, 3, 4]);

        let s5 = S5(1, 0xff_00_00_ff);
        let transd : [u8, .. 5] = cast::transmute(s5);
        // Don't worry about endianness, the u32 is palindromic.
        assert_eq!(transd, [1, 0xff, 0, 0, 0xff]);
    }
}
