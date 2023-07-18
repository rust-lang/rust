// compile-flags: -C opt-level=1

#![no_builtins]
#![crate_type = "lib"]

// CHECK: define
// CHECK-SAME: @__aeabi_memcpy
// CHECK-SAME: #0
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memcpy(dest: *mut u8, src: *const u8, size: usize) {
    // CHECK: call
    // CHECK-SAME: @memcpy(
    memcpy(dest, src, size);
}

// CHECK: declare
// CHECK-SAME: @memcpy
// CHECK-SAME: #0
extern "C" {
    pub fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8;
}

// CHECK: attributes #0
// CHECK-SAME: "no-builtins"
