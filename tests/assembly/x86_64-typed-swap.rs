//@ revisions: WIN LIN
//@ [WIN] only-windows
//@ [LIN] only-linux
//@ only-x86_64
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -O

use std::arch::x86_64::__m128;
use std::mem::swap;

// CHECK-LABEL: swap_i32:
#[no_mangle]
pub fn swap_i32(x: &mut i32, y: &mut i32) {
    // CHECK: movl (%[[ARG1:.+]]), %[[T1:.+]]
    // CHECK: movl (%[[ARG2:.+]]), %[[T2:.+]]
    // CHECK: movl %[[T2]], (%[[ARG1]])
    // CHECK: movl %[[T1]], (%[[ARG2]])
    // CHECK: retq
    swap(x, y)
}

// CHECK-LABEL: swap_pair:
#[no_mangle]
pub fn swap_pair(x: &mut (i32, u32), y: &mut (i32, u32)) {
    // CHECK: movq (%[[ARG1]]), %[[T1:.+]]
    // CHECK: movq (%[[ARG2]]), %[[T2:.+]]
    // CHECK: movq %[[T2]], (%[[ARG1]])
    // CHECK: movq %[[T1]], (%[[ARG2]])
    // CHECK: retq
    swap(x, y)
}

// CHECK-LABEL: swap_str:
#[no_mangle]
pub fn swap_str<'a>(x: &mut &'a str, y: &mut &'a str) {
    // CHECK: movups (%[[ARG1]]), %[[T1:xmm.]]
    // CHECK: movups (%[[ARG2]]), %[[T2:xmm.]]
    // CHECK: movups %[[T2]], (%[[ARG1]])
    // CHECK: movups %[[T1]], (%[[ARG2]])
    // CHECK: retq
    swap(x, y)
}

// CHECK-LABEL: swap_simd:
#[no_mangle]
pub fn swap_simd(x: &mut __m128, y: &mut __m128) {
    // CHECK: movaps (%[[ARG1]]), %[[T1:xmm.]]
    // CHECK: movaps (%[[ARG2]]), %[[T2:xmm.]]
    // CHECK: movaps %[[T2]], (%[[ARG1]])
    // CHECK: movaps %[[T1]], (%[[ARG2]])
    // CHECK: retq
    swap(x, y)
}
