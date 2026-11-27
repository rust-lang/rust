// Compiletest for rust-lang/rust#150419: Do not spill operands to the stack when
// creating debuginfo for AArch64 SVE predicates `<vscale x N x i1>` where `N != 16`
//@ edition: 2021
//@ only-aarch64
//@ build-pass
//@ compile-flags: -C debuginfo=2 -C target-feature=+sve

#![crate_type = "lib"]
#![allow(internal_features)]
#![feature(rustc_attrs, link_llvm_intrinsics)]

#[rustc_scalable_vector(16)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
pub struct svbool_t(bool);

#[rustc_scalable_vector(4)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
pub struct svbool4_t(bool);

impl std::convert::Into<svbool_t> for svbool4_t {
    #[inline(always)]
    fn into(self) -> svbool_t {
        unsafe extern "C" {
            #[link_name = "llvm.aarch64.sve.convert.to.svbool.nxv4i1"]
            fn convert_to_svbool(b: svbool4_t) -> svbool_t;
        }
        unsafe { convert_to_svbool(self) }
    }
}

pub fn svwhilelt_b32_u64(op1: u64, op2: u64) -> svbool_t {
    unsafe extern "C" {
        #[link_name = "llvm.aarch64.sve.whilelo.nxv4i1.u64"]
        fn _svwhilelt_b32_u64(op1: u64, op2: u64) -> svbool4_t;
    }
    unsafe { _svwhilelt_b32_u64(op1, op2) }.into()
}
