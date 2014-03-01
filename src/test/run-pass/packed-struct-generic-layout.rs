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
struct S<T, S> {
    a: T,
    b: u8,
    c: S
}

pub fn main() {
    unsafe {
        let s = S { a: 0xff_ff_ff_ffu32, b: 1, c: 0xaa_aa_aa_aa as i32 };
        let transd : [u8, .. 9] = cast::transmute(s);
        // Don't worry about endianness, the numbers are palindromic.
        assert!(transd ==
                   [0xff, 0xff, 0xff, 0xff,
                    1,
                    0xaa, 0xaa, 0xaa, 0xaa]);


        let s = S { a: 1u8, b: 2u8, c: 0b10000001_10000001 as i16};
        let transd : [u8, .. 4] = cast::transmute(s);
        // Again, no endianness problems.
        assert!(transd ==
                   [1, 2, 0b10000001, 0b10000001]);
    }
}
