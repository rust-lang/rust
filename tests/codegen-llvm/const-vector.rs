//@ revisions: OPT0 OPT0_S390X
//@ [OPT0] ignore-s390x
//@ [OPT0_S390X] only-s390x
//@ [OPT0] compile-flags: -C no-prepopulate-passes -Copt-level=0
//@ [OPT0_S390X] compile-flags: -C no-prepopulate-passes -Copt-level=0 -C target-cpu=z13

// This test checks that constants of SIMD type are passed as immediate vectors.
// We ensure that both vector representations (struct with fields and struct wrapping array) work.
#![crate_type = "lib"]
#![feature(abi_unadjusted)]
#![feature(const_trait_impl)]
#![feature(repr_simd)]
#![feature(rustc_attrs)]
#![feature(simd_ffi)]
#![feature(arm_target_feature)]
#![feature(mips_target_feature)]
#![allow(non_camel_case_types)]
#![feature(riscv_target_feature)]

#[path = "../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::{PackedSimd as Simd, f32x2, i8x2};

// The following functions are required for the tests to ensure
// that they are called with a const vector

extern "unadjusted" {
    fn test_i8x2(a: i8x2);
    fn test_i8x2_two_args(a: i8x2, b: i8x2);
    fn test_i8x2_mixed_args(a: i8x2, c: i32, b: i8x2);
    fn test_i8x2_arr(a: i8x2);
    fn test_f32x2(a: f32x2);
    fn test_f32x2_arr(a: f32x2);
    fn test_simd(a: Simd<i32, 4>);
    fn test_simd_unaligned(a: Simd<i32, 3>);
}

// Ensure the packed variant of the simd struct does not become a const vector
// if the size is not a power of 2
// CHECK: %"minisimd::PackedSimd<i32, 3>" = type { [3 x i32] }

#[cfg_attr(target_family = "wasm", target_feature(enable = "simd128"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86", target_feature(enable = "sse"))]
#[cfg_attr(target_arch = "mips", target_feature(enable = "msa"))]
#[cfg_attr(target_arch = "riscv64", target_feature(enable = "v"))]
pub fn do_call() {
    unsafe {
        // CHECK: call void @test_i8x2(<2 x i8> <i8 32, i8 64>
        test_i8x2(const { i8x2::from_array([32, 64]) });

        // CHECK: call void @test_i8x2_two_args(<2 x i8> <i8 32, i8 64>, <2 x i8> <i8 8, i8 16>
        test_i8x2_two_args(
            const { i8x2::from_array([32, 64]) },
            const { i8x2::from_array([8, 16]) },
        );

        // CHECK: call void @test_i8x2_mixed_args(<2 x i8> <i8 32, i8 64>, i32 43, <2 x i8> <i8 8, i8 16>
        test_i8x2_mixed_args(
            const { i8x2::from_array([32, 64]) },
            43,
            const { i8x2::from_array([8, 16]) },
        );

        // CHECK: call void @test_i8x2_arr(<2 x i8> <i8 32, i8 64>
        test_i8x2_arr(const { i8x2::from_array([32, 64]) });

        // CHECK: call void @test_f32x2(<2 x float> <float 0x3FD47AE140000000, float 0x3FE47AE140000000>
        test_f32x2(const { f32x2::from_array([0.32, 0.64]) });

        // CHECK: void @test_f32x2_arr(<2 x float> <float 0x3FD47AE140000000, float 0x3FE47AE140000000>
        test_f32x2_arr(const { f32x2::from_array([0.32, 0.64]) });

        // CHECK: call void @test_simd(<4 x i32> <i32 2, i32 4, i32 6, i32 8>
        test_simd(const { Simd::<i32, 4>([2, 4, 6, 8]) });

        // CHECK: call void @test_simd_unaligned(%"minisimd::PackedSimd<i32, 3>" %1
        test_simd_unaligned(const { Simd::<i32, 3>([2, 4, 6]) });
    }
}
