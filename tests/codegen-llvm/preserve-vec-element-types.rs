// ignore-tidy-linelength
//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled --target=aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64
//@ add-minicore
#![feature(no_core, repr_simd, f16, f128)]
#![crate_type = "lib"]
#![no_std]
#![no_core]
#![allow(non_camel_case_types)]

// Test that the SIMD vector element type is preserved. This is not required for correctness, but
// useful for optimization. It prevents additional bitcasts that make LLVM patterns fail.

extern crate minicore;
use minicore::*;

#[repr(simd)]
pub struct Simd<T, const N: usize>([T; N]);

#[repr(C)]
struct Pair<T>(T, T);

#[repr(C)]
struct Triple<T>(T, T, T);

#[repr(C)]
struct Quad<T>(T, T, T, T);

#[rustfmt::skip]
mod tests {
    use super::*;

    // CHECK: define [2 x <8 x i8>] @pair_int8x8_t([2 x <8 x i8>] {{.*}} %0)
    #[unsafe(no_mangle)] extern "C" fn pair_int8x8_t(x: Pair<Simd<i8, 8>>) -> Pair<Simd<i8, 8>> { x }

    // CHECK: define [2 x <4 x i16>] @pair_int16x4_t([2 x <4 x i16>] {{.*}} %0)
    #[unsafe(no_mangle)] extern "C" fn pair_int16x4_t(x: Pair<Simd<i16, 4>>) -> Pair<Simd<i16, 4>> { x }

    // CHECK: define [2 x <2 x i32>] @pair_int32x2_t([2 x <2 x i32>] {{.*}} %0)
    #[unsafe(no_mangle)] extern "C" fn pair_int32x2_t(x: Pair<Simd<i32, 2>>) -> Pair<Simd<i32, 2>> { x }

    // CHECK: define [2 x <1 x i64>] @pair_int64x1_t([2 x <1 x i64>] {{.*}} %0)
    #[unsafe(no_mangle)] extern "C" fn pair_int64x1_t(x: Pair<Simd<i64, 1>>) -> Pair<Simd<i64, 1>> { x }

    // CHECK: define [2 x <4 x half>] @pair_float16x4_t([2 x <4 x half>] {{.*}} %0)
    #[unsafe(no_mangle)] extern "C" fn pair_float16x4_t(x: Pair<Simd<f16, 4>>) -> Pair<Simd<f16, 4>> { x }

    // CHECK: define [2 x <2 x float>] @pair_float32x2_t([2 x <2 x float>] {{.*}} %0)
    #[unsafe(no_mangle)] extern "C" fn pair_float32x2_t(x: Pair<Simd<f32, 2>>) -> Pair<Simd<f32, 2>> { x }

    // CHECK: define [2 x <1 x double>] @pair_float64x1_t([2 x <1 x double>] {{.*}} %0)
    #[unsafe(no_mangle)] extern "C" fn pair_float64x1_t(x: Pair<Simd<f64, 1>>) -> Pair<Simd<f64, 1>> { x }

    // CHECK: define [2 x <1 x ptr>] @pair_ptrx1_t([2 x <1 x ptr>] {{.*}} %0)
    #[unsafe(no_mangle)] extern "C" fn pair_ptrx1_t(x: Pair<Simd<*const (), 1>>) -> Pair<Simd<*const (), 1>> { x }

    // When it fits in a 128-bit register, it's passed directly.

    // CHECK: define [4 x <4 x i8>] @quad_int8x4_t([4 x <4 x i8>] {{.*}} %0)
    #[unsafe(no_mangle)] extern "C" fn quad_int8x4_t(x: Quad<Simd<i8, 4>>) -> Quad<Simd<i8, 4>> { x }

    // CHECK: define [4 x <2 x i16>] @quad_int16x2_t([4 x <2 x i16>] {{.*}} %0)
    #[unsafe(no_mangle)] extern "C" fn quad_int16x2_t(x: Quad<Simd<i16, 2>>) -> Quad<Simd<i16, 2>> { x }

    // CHECK: define [4 x <1 x i32>] @quad_int32x1_t([4 x <1 x i32>] {{.*}} %0)
    #[unsafe(no_mangle)] extern "C" fn quad_int32x1_t(x: Quad<Simd<i32, 1>>) -> Quad<Simd<i32, 1>> { x }

    // CHECK: define [4 x <2 x half>] @quad_float16x2_t([4 x <2 x half>] {{.*}} %0)
    #[unsafe(no_mangle)] extern "C" fn quad_float16x2_t(x: Quad<Simd<f16, 2>>) -> Quad<Simd<f16, 2>> { x }

    // CHECK: define [4 x <1 x float>] @quad_float32x1_t([4 x <1 x float>] {{.*}} %0)
    #[unsafe(no_mangle)] extern "C" fn quad_float32x1_t(x: Quad<Simd<f32, 1>>) -> Quad<Simd<f32, 1>> { x }

    // When it doesn't quite fit, padding is added which does erase the type.

    // CHECK: define [2 x i64] @triple_int8x4_t
    #[unsafe(no_mangle)] extern "C" fn triple_int8x4_t(x: Triple<Simd<i8, 4>>) -> Triple<Simd<i8, 4>> { x }

    // Other configurations are not passed by-value but indirectly.

    // CHECK: define void @pair_int128x1_t
    #[unsafe(no_mangle)] extern "C" fn pair_int128x1_t(x: Pair<Simd<i128, 1>>) -> Pair<Simd<i128, 1>> { x }

    // CHECK: define void @pair_float128x1_t
    #[unsafe(no_mangle)] extern "C" fn pair_float128x1_t(x: Pair<Simd<f128, 1>>) -> Pair<Simd<f128, 1>> { x }

    // CHECK: define void @pair_int8x16_t
    #[unsafe(no_mangle)] extern "C" fn pair_int8x16_t(x: Pair<Simd<i8, 16>>) -> Pair<Simd<i8, 16>> { x }

    // CHECK: define void @pair_int16x8_t
    #[unsafe(no_mangle)] extern "C" fn pair_int16x8_t(x: Pair<Simd<i16, 8>>) -> Pair<Simd<i16, 8>> { x }

    // CHECK: define void @triple_int16x8_t
    #[unsafe(no_mangle)] extern "C" fn triple_int16x8_t(x: Triple<Simd<i16, 8>>) -> Triple<Simd<i16, 8>> { x }

    // CHECK: define void @quad_int16x8_t
    #[unsafe(no_mangle)] extern "C" fn quad_int16x8_t(x: Quad<Simd<i16, 8>>) -> Quad<Simd<i16, 8>> { x }
}
