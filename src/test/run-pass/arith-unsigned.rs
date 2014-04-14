// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(type_limits)]

// Unsigned integer operations
pub fn main() {
    assert!((0u8 < 255u8));
    assert!((0u8 <= 255u8));
    assert!((255u8 > 0u8));
    assert!((255u8 >= 0u8));
    assert_eq!(250u8 / 10u8, 25u8);
    assert_eq!(255u8 % 10u8, 5u8);
    assert!((0u16 < 60000u16));
    assert!((0u16 <= 60000u16));
    assert!((60000u16 > 0u16));
    assert!((60000u16 >= 0u16));
    assert_eq!(60000u16 / 10u16, 6000u16);
    assert_eq!(60005u16 % 10u16, 5u16);
    assert!((0u32 < 4000000000u32));
    assert!((0u32 <= 4000000000u32));
    assert!((4000000000u32 > 0u32));
    assert!((4000000000u32 >= 0u32));
    assert_eq!(4000000000u32 / 10u32, 400000000u32);
    assert_eq!(4000000005u32 % 10u32, 5u32);
    // 64-bit numbers have some flakiness yet. Not tested

}
