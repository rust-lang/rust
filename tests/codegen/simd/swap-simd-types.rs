//@ compile-flags: -O -C target-feature=+avx
//@ only-x86_64

#![crate_type = "lib"]

use std::mem::swap;

// SIMD types are highly-aligned already, so make sure the swap code leaves their
// types alone and doesn't pessimize them (such as by swapping them as `usize`s).
extern crate core;
use core::arch::x86_64::__m256;

// CHECK-LABEL: @swap_single_m256
#[no_mangle]
pub fn swap_single_m256(x: &mut __m256, y: &mut __m256) {
    // CHECK-NOT: alloca
    // CHECK: load <8 x float>{{.+}}align 32
    // CHECK: store <8 x float>{{.+}}align 32
    swap(x, y)
}

// CHECK-LABEL: @swap_m256_slice
#[no_mangle]
pub fn swap_m256_slice(x: &mut [__m256], y: &mut [__m256]) {
    // CHECK-NOT: alloca

    // CHECK-NOT: load i128
    // CHECK-NOT: load i64
    // CHECK-NOT: load i32

    // CHECK: [[A:%.+]] = load i256{{.+}}align 32
    // CHECK: [[B:%.+]] = load i256{{.+}}align 32
    // CHECK: store i256 [[B]]{{.+}}align 32
    // CHECK: store i256 [[A]]{{.+}}align 32

    // CHECK-NOT: load i128
    // CHECK-NOT: load i64
    // CHECK-NOT: load i32
    if x.len() == y.len() {
        x.swap_with_slice(y);
    }
}

// CHECK-LABEL: @swap_bytes32
#[no_mangle]
pub fn swap_bytes32(x: &mut [u8; 32], y: &mut [u8; 32]) {
    // CHECK-NOT: alloca

    // CHECK-NOT: load i128
    // CHECK-NOT: load i64
    // CHECK-NOT: load i32

    // CHECK: [[A:%.+]] = load i256{{.+}}align 1
    // CHECK: [[B:%.+]] = load i256{{.+}}align 1
    // CHECK: store i256 [[B]]{{.+}}align 1
    // CHECK: store i256 [[A]]{{.+}}align 1

    // CHECK-NOT: load i128
    // CHECK-NOT: load i64
    // CHECK-NOT: load i32
    swap(x, y)
}
