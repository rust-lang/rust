//@ revisions: OPT0 OPT0_S390X
//@ min-llvm-version: 22
//@ [OPT0] ignore-s390x
//@ [OPT0_S390X] only-s390x
//@ [OPT0] compile-flags: -C no-prepopulate-passes -Copt-level=0
//@ [OPT0_S390X] compile-flags: -C no-prepopulate-passes -Copt-level=0 -C target-cpu=z13

// This test checks that constants of SIMD type are passed as immediate vectors.
// We ensure that both vector representations (struct with fields and struct wrapping array) work.
#![crate_type = "lib"]
#![feature(abi_unadjusted)]
#![feature(const_trait_impl)]
#![feature(link_llvm_intrinsics)]
#![feature(repr_simd)]
#![feature(rustc_attrs)]
#![feature(simd_ffi)]
#![feature(arm_target_feature)]
#![feature(mips_target_feature)]
#![allow(non_camel_case_types)]
#![feature(riscv_target_feature)]

#[path = "../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::{PackedSimd, Simd, f32x2, i8x2};

// The following functions are required for the tests to ensure
// that they are called with a const vector

extern "unadjusted" {
    #[link_name = "llvm.vector.reduce.add.v2i8"]
    fn test_i8x2(a: i8x2) -> i8;
    #[link_name = "llvm.vector.partial.reduce.add.v2i8.v2i8.v2i8"]
    fn test_i8x2_two_args(a: i8x2, b: i8x2) -> i8x2;
    #[link_name = "llvm.vector.insert.v2i8.v2i8"]
    fn test_i8x2_mixed_args(a: i8x2, b: i8x2, c: u64) -> i8x2;
    #[link_name = "llvm.vector.reduce.fadd.v2f32"]
    fn test_f32x2(a: f32, b: f32x2) -> f32;
    #[link_name = "llvm.vector.reduce.add.v4i32"]
    fn test_simd4(a: PackedSimd<i32, 4>) -> i32;
    #[link_name = "llvm.vector.reduce.add.v3i32"]
    fn test_simd3(a: Simd<i32, 3>) -> i32;
}

#[cfg_attr(target_family = "wasm", target_feature(enable = "simd128"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86", target_feature(enable = "sse"))]
#[cfg_attr(target_arch = "mips", target_feature(enable = "msa"))]
#[cfg_attr(target_arch = "riscv64", target_feature(enable = "v"))]
pub fn do_call() {
    unsafe {
        // CHECK: call i8 @llvm.vector.reduce.add.v2i8(<2 x i8> <i8 32, i8 64>
        test_i8x2(const { i8x2::from_array([32, 64]) });

        // CHECK: call <2 x i8> @llvm.vector.partial.reduce.add.v2i8.v2i8(<2 x i8> <i8 32, i8 64>, <2 x i8> <i8 8, i8 16>)
        test_i8x2_two_args(
            const { i8x2::from_array([32, 64]) },
            const { i8x2::from_array([8, 16]) },
        );

        // CHECK: call <2 x i8> @llvm.vector.insert.v2i8.v2i8(<2 x i8> <i8 32, i8 64>, <2 x i8> <i8 8, i8 16>, i64 0
        test_i8x2_mixed_args(
            const { i8x2::from_array([32, 64]) },
            const { i8x2::from_array([8, 16]) },
            0,
        );

        // CHECK: call float @llvm.vector.reduce.fadd.v2f32(float 0.000000e+00, <2 x float> <float {{0x3FD47AE140000000|3.200000e-01}}, float {{0x3FE47AE140000000|6.400000e-01}}>
        test_f32x2(0.0, const { f32x2::from_array([0.32, 0.64]) });

        // CHECK: call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> <i32 2, i32 4, i32 6, i32 8>
        test_simd4(const { PackedSimd::<i32, 4>([2, 4, 6, 8]) });

        // CHECK: call i32 @llvm.vector.reduce.add.v3i32(<3 x i32> <i32 2, i32 4, i32 6>
        test_simd3(const { Simd::<i32, 3>([2, 4, 6]) });
    }
}
