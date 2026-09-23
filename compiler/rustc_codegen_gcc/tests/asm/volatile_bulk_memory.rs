//@ assembly-output: emit-asm
//@ only-x86_64-unknown-linux-gnu
//@ compile-flags: -Copt-level=3

#![feature(core_intrinsics)]
#![crate_type = "lib"]

use std::intrinsics::{
    volatile_copy_memory, volatile_copy_nonoverlapping_memory, volatile_set_memory,
};

// The buffers below are never read back, so the writes only survive because they are volatile.
// The functions are ordered alphabetically because that is the order they are emitted in.

// CHECK-LABEL: "volatile_copy":
// CHECK: mov
#[no_mangle]
pub unsafe fn volatile_copy(source: *const u8) {
    let mut buffer = [1u8; 64];
    volatile_copy_memory(buffer.as_mut_ptr(), source, 64);
}

// CHECK-LABEL: "volatile_copy_nonoverlapping":
// CHECK: mov
#[no_mangle]
pub unsafe fn volatile_copy_nonoverlapping(source: *const u8) {
    let mut buffer = [1u8; 64];
    volatile_copy_nonoverlapping_memory(buffer.as_mut_ptr(), source, 64);
}

// CHECK-LABEL: "volatile_set":
// CHECK: mov
#[no_mangle]
pub unsafe fn volatile_set() {
    let mut buffer = [1u8; 64];
    volatile_set_memory(buffer.as_mut_ptr(), 0, 64);
}
