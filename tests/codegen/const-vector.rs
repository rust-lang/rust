//@ compile-flags: -C no-prepopulate-passes -Copt-level=0

// This test checks that constants of SIMD type are passed as immediate vectors.
// We ensure that both vector representations (struct with fields and struct wrapping array) work.
#![crate_type = "lib"]
#![feature(abi_unadjusted)]
#![feature(const_trait_impl)]
#![feature(repr_simd)]
#![feature(rustc_attrs)]
#![feature(simd_ffi)]
#![allow(non_camel_case_types)]

// Setting up structs that can be used as const vectors
#[repr(simd)]
#[derive(Clone)]
pub struct i8x2([i8; 2]);

#[repr(simd)]
#[derive(Clone)]
pub struct f32x2([f32; 2]);

#[repr(simd, packed)]
#[derive(Copy, Clone)]
pub struct Simd<T, const N: usize>([T; N]);

// The following functions are required for the tests to ensure
// that they are called with a const vector

extern "unadjusted" {
    fn test_i8x2(a: i8x2);
}

extern "unadjusted" {
    fn test_i8x2_two_args(a: i8x2, b: i8x2);
}

extern "unadjusted" {
    fn test_i8x2_mixed_args(a: i8x2, c: i32, b: i8x2);
}

extern "unadjusted" {
    fn test_i8x2_arr(a: i8x2);
}

extern "unadjusted" {
    fn test_f32x2(a: f32x2);
}

extern "unadjusted" {
    fn test_f32x2_arr(a: f32x2);
}

extern "unadjusted" {
    fn test_simd(a: Simd<i32, 4>);
}

extern "unadjusted" {
    fn test_simd_unaligned(a: Simd<i32, 3>);
}

// Ensure the packed variant of the simd struct does not become a const vector
// if the size is not a power of 2
// CHECK: %"Simd<i32, 3>" = type { [3 x i32] }

pub fn do_call() {
    unsafe {
        // CHECK: call void @test_i8x2(<2 x i8> <i8 32, i8 64>
        test_i8x2(const { i8x2([32, 64]) });

        // CHECK: call void @test_i8x2_two_args(<2 x i8> <i8 32, i8 64>, <2 x i8> <i8 8, i8 16>
        test_i8x2_two_args(const { i8x2([32, 64]) }, const { i8x2([8, 16]) });

        // CHECK: call void @test_i8x2_mixed_args(<2 x i8> <i8 32, i8 64>, i32 43, <2 x i8> <i8 8, i8 16>
        test_i8x2_mixed_args(const { i8x2([32, 64]) }, 43, const { i8x2([8, 16]) });

        // CHECK: call void @test_i8x2_arr(<2 x i8> <i8 32, i8 64>
        test_i8x2_arr(const { i8x2([32, 64]) });

        // CHECK: call void @test_f32x2(<2 x float> <float 0x3FD47AE140000000, float 0x3FE47AE140000000>
        test_f32x2(const { f32x2([0.32, 0.64]) });

        // CHECK: void @test_f32x2_arr(<2 x float> <float 0x3FD47AE140000000, float 0x3FE47AE140000000>
        test_f32x2_arr(const { f32x2([0.32, 0.64]) });

        // CHECK: call void @test_simd(<4 x i32> <i32 2, i32 4, i32 6, i32 8>
        test_simd(const { Simd::<i32, 4>([2, 4, 6, 8]) });

        // CHECK: call void @test_simd_unaligned(%"Simd<i32, 3>" %1
        test_simd_unaligned(const { Simd::<i32, 3>([2, 4, 6]) });
    }
}
