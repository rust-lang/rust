// run-pass

#![feature(core_intrinsics)]

use std::intrinsics;

const SWAPPED_U8: u8 = intrinsics::bswap(0x12_u8);
const SWAPPED_U16: u16 = intrinsics::bswap(0x12_34_u16);
const SWAPPED_I32: i32 = intrinsics::bswap(0x12_34_56_78_i32);

fn main() {
    assert_eq!(SWAPPED_U8, 0x12);
    assert_eq!(SWAPPED_U16, 0x34_12);
    assert_eq!(SWAPPED_I32, 0x78_56_34_12);
}
