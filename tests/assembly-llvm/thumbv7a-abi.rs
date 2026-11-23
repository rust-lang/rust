//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 --target thumbv7a-pc-windows-msvc
//@ needs-llvm-components: arm

// Test that the right ABI is used for thumb windows targets.

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(C)]
pub struct Pair {
    pub a: f32,
    pub b: f32,
}

// CHECK-LABEL: make_pair
#[unsafe(no_mangle)]
pub extern "C" fn make_pair(a: f32, b: f32) -> Pair {
    // CHECK: vmov.f32 s2, s0
    // CHECK-NEXT: vmov.f32 s0, s1
    // CHECK-NEXT: vmov.f32 s1, s2
    // CHECK-NEXT: bx lr
    Pair { a: b, b: a }
}
