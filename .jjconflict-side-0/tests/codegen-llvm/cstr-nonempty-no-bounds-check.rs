//@ compile-flags: -Copt-level=3 -Cpanic=abort

#![crate_type = "lib"]

use std::ffi::CStr;

// A `CStr` always contains at least its trailing nul byte.

// CHECK-LABEL: @cstr_with_nul_is_nonempty
#[no_mangle]
pub fn cstr_with_nul_is_nonempty(s: &CStr) -> bool {
    // CHECK: ret i1 true
    !s.to_bytes_with_nul().is_empty()
}

// CHECK-LABEL: @cstr_first_with_nul
#[no_mangle]
pub fn cstr_first_with_nul(s: &CStr) -> u8 {
    // CHECK-NOT: panic_bounds_check
    // CHECK: ret i8
    s.to_bytes_with_nul()[0]
}
