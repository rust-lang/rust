//@ build-pass
//@ compile-flags: --crate-type=lib
//@ only-aarch64
#![allow(internal_features)]
#![feature(
    link_llvm_intrinsics,
    rustc_attrs,
    simd_ffi,
)]

#[derive(Copy, Clone)]
#[rustc_scalable_vector(4)]
#[allow(non_camel_case_types)]
pub struct svint32_t(i32);

#[target_feature(enable = "sve")]
pub unsafe fn svdup_n_s32(op: i32) -> svint32_t {
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.dup.x.nxv4i32")]
        fn _svdup_n_s32(op: i32) -> svint32_t;
//~^ WARN: `extern` block uses type `svint32_t`, which is not FFI-safe
    }
    unsafe { _svdup_n_s32(op) }
}

// Tests that scalable vectors can be locals, arguments and return types.

#[target_feature(enable = "sve")]
fn id(v: svint32_t) -> svint32_t { v }

#[target_feature(enable = "sve")]
fn foo() {
    unsafe {
        let v = svdup_n_s32(1);
        let v = id(v);
    }
}
