#![crate_type = "lib"]
#![feature(core_intrinsics)]

pub fn mask_ptr(ptr: *const u8, mask: usize) -> *const u8 {
    // CHECK: llvm.ptrmask
    core::intrinsics::ptr_mask(ptr, mask)
}
