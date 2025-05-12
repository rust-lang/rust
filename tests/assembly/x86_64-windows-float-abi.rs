//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3
//@ compile-flags: --target x86_64-pc-windows-msvc
//@ needs-llvm-components: x86
//@ add-core-stubs

#![feature(f16, f128)]
#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: second_f16
// CHECK: movaps %xmm1, %xmm0
// CHECK-NEXT: retq
#[no_mangle]
pub extern "C" fn second_f16(_: f16, x: f16) -> f16 {
    x
}

// CHECK-LABEL: second_f32
// CHECK: movaps %xmm1, %xmm0
// CHECK-NEXT: retq
#[no_mangle]
pub extern "C" fn second_f32(_: f32, x: f32) -> f32 {
    x
}

// CHECK-LABEL: second_f64
// CHECK: movaps %xmm1, %xmm0
// CHECK-NEXT: retq
#[no_mangle]
pub extern "C" fn second_f64(_: f64, x: f64) -> f64 {
    x
}

// CHECK-LABEL: second_f128
// FIXME(llvm21): this can be just %rdx instead of the regex once we don't test on LLVM 20
// CHECK: movaps {{(%xmm1|\(%rdx\))}}, %xmm0
// CHECK-NEXT: retq
#[no_mangle]
pub extern "C" fn second_f128(_: f128, x: f128) -> f128 {
    x
}
