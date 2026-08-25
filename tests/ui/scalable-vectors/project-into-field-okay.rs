//@ build-pass
//@ compile-flags: -Copt-level=3 --test -Cdebuginfo=2
//@ only-aarch64
#![allow(internal_features, unused, improper_ctypes_definitions, nonstandard_style)]
#![feature(
    abi_unadjusted,
    link_llvm_intrinsics,
    rustc_attrs,
    min_adt_const_params
)]

// This snippet is reduced from stdarch and previously failed prior to preventing projection into
// fields during MIR validation in rust#160642.

use std::{marker::ConstParamTy, mem::transmute};

#[derive(Copy, Clone)]
#[rustc_scalable_vector(4)]
pub struct svfloat32_t(f32);

#[repr(i32)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
#[non_exhaustive]
pub enum svpattern {
    SV_ALL = 31,
}

#[inline]
#[target_feature(enable = "sve")]
pub fn svlen_f32(_op: svfloat32_t) -> u64 {
    svcntw()
}

#[test]
#[allow(non_snake_case)]
fn assert_svlen_f32_cntw() {
    #[target_feature(enable = "sve")]
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub unsafe extern "C" fn stdarch_test_shim_svlen_f32_cntw(_op: svfloat32_t) -> u64 {
        svlen_f32(_op)
    }
    std::hint::black_box(stdarch_test_shim_svlen_f32_cntw as usize);
    //~^ WARN: direct cast of function item into an integer
}

#[inline]
#[target_feature(enable = "sve")]
pub fn svcntw() -> u64 {
    svcntw_pat::<{ svpattern::SV_ALL }>()
}

#[test]
#[allow(non_snake_case)]
fn assert_svcntw_cntw() {
    #[target_feature(enable = "sve")]
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub unsafe extern "C" fn stdarch_test_shim_svcntw_cntw() -> u64 {
        svcntw()
    }
    std::hint::black_box(stdarch_test_shim_svcntw_cntw as usize);
    //~^ WARN: direct cast of function item into an integer
}

#[inline]
#[target_feature(enable = "sve")]
pub fn svcntw_pat<const PATTERN: svpattern>() -> u64 {
    unsafe extern "unadjusted" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cntw")]
        fn _svcntw_pat(pattern: svpattern) -> i64;
    }
    unsafe { transmute(_svcntw_pat(PATTERN)) }
    //~^ WARN: unnecessary transmute
}

#[test]
#[allow(non_snake_case)]
fn assert_svcntw_pat_cntw() {
    #[target_feature(enable = "sve")]
    #[unsafe(no_mangle)]
    #[inline(never)]
    pub unsafe extern "C" fn stdarch_test_shim_svcntw_pat_cntw() -> u64 {
        svcntw_pat::<{ svpattern::SV_ALL }>()
    }
    std::hint::black_box(stdarch_test_shim_svcntw_pat_cntw as usize);
    //~^ WARN: direct cast of function item into an integer
}

fn main() {
    unsafe {
        let _ = svlen_f32(unimplemented!());
    }
}
