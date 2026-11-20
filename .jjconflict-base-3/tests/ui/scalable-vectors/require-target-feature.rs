//@ build-fail
//@ compile-flags: --crate-type=lib --target=aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64
//@ add-minicore
//@ dont-require-annotations: NOTE

#![allow(incomplete_features, internal_features)]
#![feature(
    simd_ffi,
    rustc_attrs,
    link_llvm_intrinsics,
    no_core
)]
#![no_core]

extern crate minicore;
use minicore::*;

#[rustc_scalable_vector(4)]
#[allow(non_camel_case_types)]
pub struct svint32_t(i32);

impl Copy for svint32_t {}

#[inline(never)]
#[target_feature(enable = "sve")]
pub unsafe fn svdup_n_s32(op: i32) -> svint32_t {
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.dup.x.nxv4i32")]
        fn _svdup_n_s32(op: i32) -> svint32_t;
//~^ WARN: `extern` block uses type `svint32_t`, which is not FFI-safe
    }
    unsafe { _svdup_n_s32(op) }
}

pub fn non_annotated_callee(x: svint32_t) {}
//~^ ERROR: this function definition requires the `sve` target feature
//~| NOTE: the function has type `svint32_t` in its signature, which is passed in scalable vector registers under the "Rust" ABI

#[target_feature(enable = "sve")]
pub fn annotated_callee(x: svint32_t) {} // okay!

#[target_feature(enable = "sve")]
pub fn caller() {
    unsafe {
        let a = svdup_n_s32(42);
        non_annotated_callee(a);
        annotated_callee(a);
    }
}
