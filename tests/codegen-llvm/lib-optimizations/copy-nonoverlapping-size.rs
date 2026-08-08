//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

use std::ptr;

// CHECK-LABEL: @copy_size_is_valid
#[no_mangle]
pub unsafe fn copy_size_is_valid(src: *const u8, dst: *mut u8, count: usize) -> bool {
    // SAFETY: guaranteed by the caller.
    unsafe { ptr::copy_nonoverlapping(src, dst, count) };

    // CHECK: ret i1 true
    count <= isize::MAX as usize
}
