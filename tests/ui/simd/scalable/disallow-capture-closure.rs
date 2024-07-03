//@ only-aarch64

#![allow(incomplete_features, internal_features, improper_ctypes)]
#![feature(
    repr_simd,
    repr_scalable,
    simd_ffi,
    unsized_locals,
    unsized_fn_params,
    link_llvm_intrinsics
)]

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
fn run(f: impl Fn() -> ()) {
    f();
}

fn main() {
    unsafe {
        let a = svdup_n_s32(42);
        run(move || {
            svxar_n_s32::<2>(a, a); //~ ERROR E0277
        });
    }
}
