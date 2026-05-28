//@ check-pass
//@ only-aarch64
#![crate_type = "lib"]
#![allow(incomplete_features, internal_features, improper_ctypes)]
#![feature(abi_unadjusted, core_intrinsics, link_llvm_intrinsics, rustc_attrs)]

use std::intrinsics::simd::simd_select;

#[derive(Copy, Clone)]
#[rustc_scalable_vector(16)]
#[allow(non_camel_case_types)]
pub struct svbool_t(bool);

#[derive(Copy, Clone)]
#[rustc_scalable_vector(16)]
#[allow(non_camel_case_types)]
pub struct svint8_t(i8);

#[target_feature(enable = "sve")]
pub fn svsel_s8(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    unsafe { simd_select::<svbool_t, _>(pg, op1, op2) }
}
