//@ check-pass
//@ only-aarch64
#![crate_type = "lib"]
#![allow(incomplete_features, internal_features, improper_ctypes)]
#![feature(abi_unadjusted, core_intrinsics, link_llvm_intrinsics, rustc_attrs)]

use std::intrinsics::simd::scalable::sve_cast;

#[derive(Copy, Clone)]
#[rustc_scalable_vector(16)]
#[allow(non_camel_case_types)]
pub struct svbool_t(bool);

#[derive(Copy, Clone)]
#[rustc_scalable_vector(2)]
#[allow(non_camel_case_types)]
pub struct svbool2_t(bool);

#[derive(Copy, Clone)]
#[rustc_scalable_vector(2)]
#[allow(non_camel_case_types)]
pub struct svint64_t(i64);

#[derive(Copy, Clone)]
#[rustc_scalable_vector(2)]
#[allow(non_camel_case_types)]
pub struct nxv2i16(i16);

pub trait SveInto<T>: Sized {
    unsafe fn sve_into(self) -> T;
}

impl SveInto<svbool2_t> for svbool_t {
    #[target_feature(enable = "sve")]
    unsafe fn sve_into(self) -> svbool2_t {
        unsafe extern "C" {
            #[cfg_attr(
                target_arch = "aarch64",
                link_name = concat!("llvm.aarch64.sve.convert.from.svbool.nxv2i1")
            )]
            fn convert_from_svbool(b: svbool_t) -> svbool2_t;
        }
        unsafe { convert_from_svbool(self) }
    }
}

#[target_feature(enable = "sve")]
pub unsafe fn svld1sh_gather_s64offset_s64(
    pg: svbool_t,
    base: *const i16,
    offsets: svint64_t,
) -> svint64_t {
    unsafe extern "unadjusted" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.ld1.gather.nxv2i16"
        )]
        fn _svld1sh_gather_s64offset_s64(
            pg: svbool2_t,
            base: *const i16,
            offsets: svint64_t,
        ) -> nxv2i16;
    }
    sve_cast(_svld1sh_gather_s64offset_s64(pg.sve_into(), base, offsets))
}
