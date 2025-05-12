//@ assembly-output: ptx-linker
//@ compile-flags: --crate-type cdylib
//@ only-nvptx64
//@ ignore-nvptx64

#![feature(abi_ptx)]
#![no_std]

//@ aux-build: breakpoint-panic-handler.rs
extern crate breakpoint_panic_handler;

//@ aux-build: non-inline-dependency.rs
extern crate non_inline_dependency as dep;

// Verify that no extra function declarations are present.
// CHECK-NOT: .func

// CHECK: .visible .entry top_kernel(
#[no_mangle]
pub unsafe extern "ptx-kernel" fn top_kernel(a: *const u32, b: *mut u32) {
    // CHECK: add.s32 %{{r[0-9]+}}, %{{r[0-9]+}}, 5;
    *b = *a + 5;
}

// Verify that no extra function definitions are here.
// CHECK-NOT: .func
// CHECK-NOT: .entry
