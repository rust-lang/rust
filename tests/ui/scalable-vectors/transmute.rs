//@ build-pass
//@ compile-flags: -Copt-level=3
//@ only-aarch64
#![crate_type = "lib"]
#![allow(incomplete_features, internal_features, dead_code, improper_ctypes)]
#![allow(nonstandard_style, private_interfaces)]
#![feature(abi_unadjusted, link_llvm_intrinsics, rustc_attrs)]

// Tests that use of transmute between `svuint8x2_t` and `svint8x2_t` builds with optimisations
// without any failures from LLVM.

use std::mem::transmute;

#[rustc_scalable_vector(16)]
struct svbool_t(bool);

#[rustc_scalable_vector(16)]
struct svuint8_t(u8);

#[rustc_scalable_vector]
struct svuint8x2_t(svuint8_t, svuint8_t);

#[rustc_scalable_vector(16)]
struct svint8_t(i8);

#[rustc_scalable_vector]
struct svint8x2_t(svint8_t, svint8_t);

#[target_feature(enable = "sve")]
pub unsafe fn svld2_u8(pg: svbool_t, base: *const i8) -> svuint8x2_t {
    unsafe extern "unadjusted" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.ld2.sret.nxv16i8"
        )]
        fn _svld2_s8(pg: svbool_t, base: *const i8) -> svint8x2_t;
    }
    transmute(_svld2_s8(pg, base))
}
