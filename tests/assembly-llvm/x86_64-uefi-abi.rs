// Checks that `x86_64-unknown-uefi`, which is lowered through LLVM's dedicated
// UEFI target, keeps the Win64 calling convention the UEFI specification
// mandates, both for Rust functions and for the libcalls LLVM emits.

//@ assembly-output: emit-asm
//@ add-minicore
//@ compile-flags: --target x86_64-unknown-uefi -Copt-level=3
//@ needs-llvm-components: x86

#![feature(no_core, intrinsics)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[rustc_intrinsic]
unsafe fn unchecked_div<T: Copy>(x: T, y: T) -> T;

// The first four integer arguments are passed in rcx, rdx, r8 and r9, the
// fifth one lives on the stack behind the return address and the 32 bytes of
// shadow space.
// CHECK-LABEL: fifth_argument:
// CHECK: movq 40(%rsp), %rax
// CHECK-NEXT: retq
#[no_mangle]
pub extern "C" fn fifth_argument(_a: i64, _b: i64, _c: i64, _d: i64, e: i64) -> i64 {
    e
}

// 128-bit division is a libcall. On Win64 the operands are passed by pointer
// (see `LowerWIN64_i128OP` in LLVM), which is also how compiler-builtins is
// compiled for this target.
// CHECK-LABEL: udiv128:
// CHECK-DAG: leaq {{[0-9]+}}(%rsp), %rcx
// CHECK-DAG: leaq {{[0-9]+}}(%rsp), %rdx
// CHECK: callq __udivti3
#[no_mangle]
pub extern "C" fn udiv128(a: &u128, b: &u128) -> u128 {
    unsafe { unchecked_div(*a, *b) }
}
