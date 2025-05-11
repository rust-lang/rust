//@ add-core-stubs
//@ build-pass
//@ revisions: arm
//@[arm] compile-flags: --target arm-unknown-linux-gnueabi
//@[arm] needs-llvm-components: arm
//@ revisions: aarch64
//@[aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@[aarch64] needs-llvm-components: aarch64
#![feature(
    no_core, lang_items, link_llvm_intrinsics,
    abi_unadjusted, repr_simd, arm_target_feature,
)]
#![no_std]
#![no_core]
#![crate_type = "lib"]
#![allow(non_camel_case_types)]

extern crate minicore;
use minicore::*;

// Regression test for https://github.com/rust-lang/rust/issues/118124.

#[repr(simd)]
pub struct int8x16_t(pub(crate) [i8; 16]);
impl Copy for int8x16_t {}

#[repr(C)]
pub struct int8x16x4_t(pub int8x16_t, pub int8x16_t, pub int8x16_t, pub int8x16_t);
impl Copy for int8x16x4_t {}

#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
pub unsafe fn vld1q_s8_x4(a: *const i8) -> int8x16x4_t {
    #[allow(improper_ctypes)]
    extern "unadjusted" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vld1x4.v16i8.p0i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.ld1x4.v16i8.p0i8")]
        fn vld1q_s8_x4_(a: *const i8) -> int8x16x4_t;
    }
    vld1q_s8_x4_(a)
}
