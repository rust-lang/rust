// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

fn main() {
    let error = 255i8; //~WARNING literal out of range for i8

    let ok = 0b1000_0001; // should be ok -> i32
    let ok = 0b0111_1111i8; // should be ok -> 127i8

    let fail = 0b1000_0001i8; //~WARNING literal out of range for i8

    let fail = 0x8000_0000_0000_0000i64; //~WARNING literal out of range for i64

    let fail = 0x1_FFFF_FFFFu32; //~WARNING literal out of range for u32

    let fail: i128 = 0x8000_0000_0000_0000_0000_0000_0000_0000;
    //~^ WARNING literal out of range for i128

    let fail = 0x8FFF_FFFF_FFFF_FFFE; //~WARNING literal out of range for i32

    let fail = -0b1111_1111i8; //~WARNING literal out of range for i8
}
