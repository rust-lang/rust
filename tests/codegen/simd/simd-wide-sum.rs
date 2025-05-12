//@ revisions: llvm mir-opt3
//@ compile-flags: -C opt-level=3 -Z merge-functions=disabled
//@ edition: 2021
//@ only-x86_64
//@ [mir-opt3]compile-flags: -Zmir-opt-level=3
//@ [mir-opt3]build-pass

// mir-opt3 is a regression test for https://github.com/rust-lang/rust/issues/98016

#![crate_type = "lib"]
#![feature(portable_simd)]

use std::simd::prelude::*;
const N: usize = 16;

#[no_mangle]
// CHECK-LABEL: @wider_reduce_simd
pub fn wider_reduce_simd(x: Simd<u8, N>) -> u16 {
    // CHECK: zext <16 x i8>
    // CHECK-SAME: to <16 x i16>
    // CHECK: call i16 @llvm.vector.reduce.add.v16i16(<16 x i16>
    let x: Simd<u16, N> = x.cast();
    x.reduce_sum()
}

#[no_mangle]
// CHECK-LABEL: @wider_reduce_loop
pub fn wider_reduce_loop(x: Simd<u8, N>) -> u16 {
    // CHECK: zext <16 x i8>
    // CHECK-SAME: to <16 x i16>
    // CHECK: call i16 @llvm.vector.reduce.add.v16i16(<16 x i16>
    let mut sum = 0_u16;
    for i in 0..N {
        sum += u16::from(x[i]);
    }
    sum
}

#[no_mangle]
// CHECK-LABEL: @wider_reduce_iter
pub fn wider_reduce_iter(x: Simd<u8, N>) -> u16 {
    // CHECK: zext <16 x i8>
    // CHECK-SAME: to <16 x i16>
    // CHECK: call i16 @llvm.vector.reduce.add.v16i16(<16 x i16>
    x.as_array().iter().copied().map(u16::from).sum()
}

// This iterator one is the most interesting, as it's the one
// which used to not auto-vectorize due to a suboptimality in the
// `<array::IntoIter as Iterator>::fold` implementation.

#[no_mangle]
// CHECK-LABEL: @wider_reduce_into_iter
pub fn wider_reduce_into_iter(x: Simd<u8, N>) -> u16 {
    // CHECK: zext <16 x i8>
    // CHECK-SAME: to <16 x i16>
    // CHECK: call i16 @llvm.vector.reduce.add.v16i16(<16 x i16>
    x.to_array().into_iter().map(u16::from).sum()
}
