//@ only-aarch64
//@ compile-flags: -C opt-level=2 --edition=2021

#![crate_type = "lib"]
#![allow(incomplete_features, internal_features, improper_ctypes)]
#![feature(repr_simd, repr_scalable, simd_ffi, link_llvm_intrinsics)]

#[repr(simd, scalable(4))]
#[allow(non_camel_case_types)]
pub struct svint32_t {
    _ty: [i32],
}

#[inline(never)]
#[target_feature(enable = "sve")]
pub unsafe fn svdup_n_s32(op: i32) -> svint32_t {
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.dup.x.nxv4i32")]
        fn _svdup_n_s32(op: i32) -> svint32_t;
    }
    unsafe { _svdup_n_s32(op) }
}

#[inline]
#[target_feature(enable = "sve,sve2")]
pub unsafe fn svxar_n_s32<const IMM3: i32>(op1: svint32_t, op2: svint32_t) -> svint32_t {
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.xar.nxv4i32")]
        fn _svxar_n_s32(op1: svint32_t, op2: svint32_t, imm3: i32) -> svint32_t;
    }
    unsafe { _svxar_n_s32(op1, op2, IMM3) }
}

#[inline(never)]
#[no_mangle]
#[target_feature(enable = "sve,sve2")]
// CHECK: define <vscale x 4 x i32> @pass_as_ref(ptr noalias nocapture noundef readonly align 4 dereferenceable(4) %a, <vscale x 4 x i32> %b)
pub unsafe fn pass_as_ref(a: &svint32_t, b: svint32_t) -> svint32_t {
    // CHECK: load <vscale x 4 x i32>, ptr %a, align 4
    svxar_n_s32::<1>(*a, b)
}

#[no_mangle]
#[target_feature(enable = "sve,sve2")]
// CHECK: define <vscale x 4 x i32> @test()
pub unsafe fn test() -> svint32_t {
    let a = svdup_n_s32(1);
    let b = svdup_n_s32(2);
    // CHECK: call <vscale x 4 x i32> @pass_as_ref(ptr noalias noundef nonnull readonly align 4 dereferenceable(4) %a, <vscale x 4 x i32> %b)
    pass_as_ref(&a, b)
}
