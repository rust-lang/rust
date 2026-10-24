//@ compile-flags: --crate-type=lib
//@ only-aarch64

#![allow(incomplete_features, internal_features)]
#![feature(
    link_llvm_intrinsics,
    rustc_attrs,
    simd_ffi
)]

#[derive(Copy, Clone)]
#[rustc_scalable_vector(4)]
#[allow(non_camel_case_types)]
pub struct svint32_t(i32);

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
#[target_feature(enable = "sve,sve2")]
fn run(f: impl Fn() -> ()) {
    f();
}

#[target_feature(enable = "sve,sve2")]
fn foo() {
    unsafe {
        let a = svdup_n_s32(42);
        run(move || {
//~^ ERROR: scalable vectors cannot be tuple fields
            svxar_n_s32::<2>(a, a);
        });
    }
}
