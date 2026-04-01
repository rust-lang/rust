//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics;

// CHECK-LABEL: @volatile_copy_memory
#[no_mangle]
pub unsafe fn volatile_copy_memory(a: *mut u8, b: *const u8) {
    // CHECK: llvm.memmove.{{\w*(.*true)}}
    intrinsics::volatile_copy_memory(a, b, 1)
}

// CHECK-LABEL: @volatile_copy_nonoverlapping_memory
#[no_mangle]
pub unsafe fn volatile_copy_nonoverlapping_memory(a: *mut u8, b: *const u8) {
    // CHECK: llvm.memcpy.{{\w*(.*true)}}
    intrinsics::volatile_copy_nonoverlapping_memory(a, b, 1)
}

// CHECK-LABEL: @volatile_set_memory
#[no_mangle]
pub unsafe fn volatile_set_memory(a: *mut u8, b: u8) {
    // CHECK: llvm.memset.{{\w*(.*true)}}
    intrinsics::volatile_set_memory(a, b, 1)
}

// CHECK-LABEL: @volatile_load
#[no_mangle]
pub unsafe fn volatile_load(a: *const u8) -> u8 {
    // CHECK: load volatile
    intrinsics::volatile_load(a)
}

// CHECK-LABEL: @volatile_store
#[no_mangle]
pub unsafe fn volatile_store(a: *mut u8, b: u8) {
    // CHECK: store volatile
    intrinsics::volatile_store(a, b)
}

// CHECK-LABEL: @unaligned_volatile_load
#[no_mangle]
pub unsafe fn unaligned_volatile_load(a: *const u8) -> u8 {
    // CHECK: load volatile
    intrinsics::unaligned_volatile_load(a)
}

// CHECK-LABEL: @unaligned_volatile_store
#[no_mangle]
pub unsafe fn unaligned_volatile_store(a: *mut u8, b: u8) {
    // CHECK: store volatile
    intrinsics::unaligned_volatile_store(a, b)
}
