//@ only-aarch64
//@ edition:2021

#![allow(incomplete_features, internal_features)]
#![feature(
    core_intrinsics,
    simd_ffi,
    rustc_attrs,
    link_llvm_intrinsics
)]

use core::intrinsics::transmute_unchecked;

#[rustc_scalable_vector(4)]
#[allow(non_camel_case_types)]
pub struct svint32_t(i32);

#[target_feature(enable = "sve")]
pub unsafe fn svdup_n_s32(op: i32) -> svint32_t {
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.dup.x.nxv4i32")]
        fn _svdup_n_s32(op: i32) -> svint32_t;
    }
    unsafe { _svdup_n_s32(op) }
}

#[target_feature(enable = "sve")]
async fn another() -> i32 {
    42
}

#[no_mangle]
#[target_feature(enable = "sve")]
pub async fn test_function() {
    unsafe {
        let x = svdup_n_s32(1); //~ ERROR: scalable vectors cannot be held over await points
        let temp = another().await;
        let y: svint32_t = transmute_unchecked(x);
    }
}

fn main() {
    let _ = unsafe { test_function() };
}
