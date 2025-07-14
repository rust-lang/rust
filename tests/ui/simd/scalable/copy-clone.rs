//@ build-pass
//@ only-aarch64
#![feature(link_llvm_intrinsics, repr_simd, repr_scalable, simd_ffi)]

#[derive(Copy, Clone)]
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

#[target_feature(enable = "sve")]
fn require_copy<T: Copy>(t: T) {}

#[target_feature(enable = "sve")]
fn test() {
    unsafe {
        let a = svdup_n_s32(1);
        require_copy(a);
    }
}

fn main() {}
