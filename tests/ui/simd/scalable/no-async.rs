//@ only-aarch64
//@ edition:2021

#![allow(incomplete_features, internal_features, improper_ctypes)]
#![feature(
    core_intrinsics,
    repr_simd,
    repr_scalable,
    simd_ffi,
    link_llvm_intrinsics
)]

use core::intrinsics::simd::simd_reinterpret;

#[repr(simd, scalable(4))]
#[allow(non_camel_case_types)]
pub struct svint32_t {
    _ty: [i32],
}

#[target_feature(enable = "sve")]
pub unsafe fn svdup_n_s32(op: i32) -> svint32_t {
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.dup.x.nxv4i32")]
        fn _svdup_n_s32(op: i32) -> svint32_t;
    }
    unsafe { _svdup_n_s32(op) }
}

async fn another() -> i32 {
    42
}

#[no_mangle]
pub async fn test_function() {
    unsafe {
        let x = svdup_n_s32(1); //~ ERROR E0277
        let temp = another().await;
        let y: svint32_t = simd_reinterpret(x);
    }
}

fn main() {
    let _ = test_function();
}
