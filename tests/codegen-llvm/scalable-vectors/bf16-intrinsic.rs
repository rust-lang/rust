//@ add-minicore
//@ revisions: OPT-0 OPT-3
//@[OPT-0] compile-flags: -Copt-level=0
//@[OPT-3] compile-flags: -Copt-level=3
//@ compile-flags: --target aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64

#![crate_type = "lib"]
#![feature(f16b, link_llvm_intrinsics, no_core, simd_ffi)]
#![no_core]
#![allow(non_camel_case_types)]

extern crate minicore;

use minicore::num::f16b;
use minicore::simd::{Simd, f32x4};

type bfloat16x8_t = Simd<f16b, 8>;

#[unsafe(no_mangle)]
#[target_feature(enable = "neon,bf16")]
// CHECK-LABEL: define <4 x float> @vbfmmlaq_f32(
// CHECK-SAME: <4 x float> %acc, <8 x bfloat> %lhs, <8 x bfloat> %rhs)
pub unsafe extern "C" fn vbfmmlaq_f32(acc: f32x4, lhs: bfloat16x8_t, rhs: bfloat16x8_t) -> f32x4 {
    unsafe extern "llvm-intrinsic" {
        #[link_name = "llvm.aarch64.neon.bfmmla"]
        fn bfmmla(acc: f32x4, lhs: bfloat16x8_t, rhs: bfloat16x8_t) -> f32x4;
    }

    // CHECK: [[RESULT:%.*]] = {{.*}}call <4 x float> @llvm.aarch64.neon.bfmmla(
    // CHECK-SAME: <4 x float> %acc, <8 x bfloat> %lhs, <8 x bfloat> %rhs)
    // CHECK: ret <4 x float> [[RESULT]]
    unsafe { bfmmla(acc, lhs, rhs) }
}
