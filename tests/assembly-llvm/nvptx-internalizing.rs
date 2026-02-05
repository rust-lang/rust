//@ add-minicore
//@ assembly-output: ptx-linker
//@ compile-flags: --target nvptx64-nvidia-cuda --crate-type cdylib -Ctarget-cpu=sm_30
//@ needs-llvm-components: nvptx
//@ ignore-backends: gcc

#![feature(abi_ptx, no_core, intrinsics)]
#![no_core]

extern crate minicore;
use minicore::*;

#[rustc_intrinsic]
pub const unsafe fn unchecked_add<T: Copy>(x: T, y: T) -> T;

// Verify that no extra function declarations are present.
// CHECK-NOT: .func

// CHECK: .visible .entry top_kernel(
#[no_mangle]
pub unsafe extern "ptx-kernel" fn top_kernel(a: *const u32, b: *mut u32) {
    // CHECK: add.s32 %{{r[0-9]+}}, %{{r[0-9]+}}, 5;
    *b = unchecked_add(*a, 5);
}

// Verify that no extra function definitions are here.
// CHECK-NOT: .func
// CHECK-NOT: .entry
