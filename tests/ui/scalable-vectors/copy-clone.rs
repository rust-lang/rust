//@ build-pass
//@ only-aarch64
#![feature(simd_ffi, rustc_attrs, link_llvm_intrinsics)]

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
