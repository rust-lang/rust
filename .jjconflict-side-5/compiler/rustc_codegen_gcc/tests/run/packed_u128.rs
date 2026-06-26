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
#[repr(packed(1))]
pub struct ScalarInt {
    data: u128,
    size: u8,
}
#[inline(never)]
#[no_mangle]
fn read_data(a: &ScalarInt) {
    black_box(a.data);
}

#[no_mangle]
extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let data =
        [black_box(ScalarInt { data: 0, size: 1 }), black_box(ScalarInt { data: 0, size: 1 })];
    read_data(&data[1]);
    0
}
