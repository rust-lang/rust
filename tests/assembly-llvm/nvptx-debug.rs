//@ assembly-output: emit-asm
//@ compile-flags: --crate-type cdylib -C debuginfo=2
//@ only-nvptx64

// Tests related to debug symbols for nvptx

#![feature(abi_ptx)]
#![no_std]

//@ aux-build: breakpoint-panic-handler.rs
extern crate breakpoint_panic_handler;

#[no_mangle]
pub extern "ptx-kernel" fn foo() {
    panic!("bar");
}

// We make sure that all debug sections are available and visit them
// CHECK: .section .debug_abbrev
// CHECK: .section .debug_info

// Issue #99248 describes a bug where `.` was used as a seperator
// instead of `_` for `anon`s in `.debug_info`
// CHECK-NOT: anon.
