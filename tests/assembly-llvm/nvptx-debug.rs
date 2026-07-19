//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: --target=nvptx64-nvidia-cuda --crate-type cdylib -C debuginfo=2
//@ needs-llvm-components: nvptx

// Tests related to debug symbols for nvptx

#![feature(no_core, abi_ptx)]
#![no_core]

extern crate minicore;

#[no_mangle]
pub extern "ptx-kernel" fn foo() {}

// We make sure that all debug sections are available and visit them
// CHECK: .section .debug_abbrev
// CHECK: .section .debug_info

// Issue #99248 describes a bug where `.` was used as a seperator
// instead of `_` for `anon`s in `.debug_info`
// CHECK-NOT: anon.
