//@ build-pass
//@ compile-flags: --crate-type=lib
//@ only-aarch64
#![feature(link_llvm_intrinsics, simd_ffi, repr_scalable, repr_simd)]

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

// Tests that scalable vectors can be locals, arguments and return types.

fn id(v: svint32_t) -> svint32_t { v }

fn foo() {
    unsafe {
        let v = svdup_n_s32(1);
        let v = id(v);
    }
}
