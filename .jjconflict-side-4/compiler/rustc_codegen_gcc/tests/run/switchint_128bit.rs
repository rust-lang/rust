// Compiler:
//
// Run-time:
//   status: 0

#![feature(no_core)]
#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;
use intrinsics::black_box;
use mini_core::*;

#[no_mangle]
extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    // 1st. Check that small 128 bit values work.
    let val = black_box(64_u128);
    match val {
        0 => return 1,
        1 => return 2,
        64 => (),
        _ => return 3,
    }
    // 2nd check that *large* values work.
    const BIG: u128 = 0xDEAD_C0FE_BEEF_DECAF_BADD_DECAF_BEEF_u128;
    let val = black_box(BIG);
    match val {
        0 => return 4,
        1 => return 5,
        // Check that we will not match on the lower u64, if the upper qword is different!
        0xcafbadddecafbeef => return 6,
        0xDEAD_C0FE_BEEF_DECAF_BADD_DECAF_BEEF_u128 => (),
        _ => return 7,
    }
    0
}
