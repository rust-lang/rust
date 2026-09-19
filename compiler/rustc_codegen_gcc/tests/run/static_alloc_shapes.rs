// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 8
//      12
//      5
//      7
//      7
//      9

#![feature(no_core)]
#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;
use mini_core::*;

// One byte run of each length class that maps to a distinct array element type.
static mut BYTES8: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
static mut BYTES12: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
static mut BYTES5: [u8; 5] = [1, 2, 3, 4, 5];

static mut VALUE: isize = 7;
static mut OTHER: isize = 9;

// An allocation that is exactly one relocation, so it ends on a pointer with no trailing bytes.
static mut PTR: &isize = unsafe { &VALUE };

struct TwoRefs {
    first: &'static isize,
    second: &'static isize,
}

// Two adjacent relocations, with no byte run between them.
static mut TWO_REFS: TwoRefs = TwoRefs { first: unsafe { &VALUE }, second: unsafe { &OTHER } };

#[no_mangle]
extern "C" fn main(_argc: isize, _argv: *const *const u8) -> i32 {
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, BYTES8[7] as isize);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, BYTES12[11] as isize);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, BYTES5[4] as isize);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, *PTR);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, *TWO_REFS.first);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, *TWO_REFS.second);
    }
    0
}
