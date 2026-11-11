//@ compile-flags: -C opt-level=1

#![no_builtins]
#![crate_type = "lib"]

use std::ffi::c_void;

// CHECK: define
// CHECK-SAME: @__aeabi_memcpy
// CHECK-SAME: #0
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memcpy(dest: *mut c_void, src: *const c_void, size: usize) {
    // CHECK: call
    // CHECK-SAME: @memcpy(
    memcpy(dest, src, size);
}

// CHECK: declare
// CHECK-SAME: @memcpy
// CHECK-SAME: #0
extern "C" {
    pub fn memcpy(dest: *mut c_void, src: *const c_void, n: usize) -> *mut c_void;
}

// CHECK: attributes #0
// CHECK-SAME: "no-builtins"
