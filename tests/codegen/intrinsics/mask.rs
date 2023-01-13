#![crate_type = "lib"]
#![feature(core_intrinsics)]

// CHECK-LABEL: @mask_ptr
// CHECK-SAME: [[WORD:i[0-9]+]] %mask
#[no_mangle]
pub fn mask_ptr(ptr: *const u16, mask: usize) -> *const u16 {
    // CHECK: call
    // CHECK-SAME: @llvm.ptrmask.{{p0|p0i8}}.[[WORD]]({{ptr|i8\*}} {{%ptr|%0}}, [[WORD]] %mask)
    core::intrinsics::ptr_mask(ptr, mask)
}
