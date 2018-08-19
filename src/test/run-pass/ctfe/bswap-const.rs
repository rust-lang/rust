// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core_intrinsics)]

use std::intrinsics;

const SWAPPED_U8: u8 = unsafe { intrinsics::bswap(0x12_u8) };
const SWAPPED_U16: u16 = unsafe { intrinsics::bswap(0x12_34_u16) };
const SWAPPED_I32: i32 = unsafe { intrinsics::bswap(0x12_34_56_78_i32) };

fn main() {
    assert_eq!(SWAPPED_U8, 0x12);
    assert_eq!(SWAPPED_U16, 0x34_12);
    assert_eq!(SWAPPED_I32, 0x78_56_34_12);
}
