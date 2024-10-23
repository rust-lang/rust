//@ compile-flags: -O
#![crate_type = "lib"]

/// Make sure no bounds checks are emitted after a `get_unchecked`.
// CHECK-LABEL: @unchecked_slice_no_bounds_check
#[no_mangle]
pub unsafe fn unchecked_slice_no_bounds_check(s: &[u8]) -> u8 {
    let a = *s.get_unchecked(1);
    // CHECK-NOT: panic_bounds_check
    a + s[0]
}

// CHECK-LABEL: @unchecked_slice_no_bounds_check_mut
#[no_mangle]
pub unsafe fn unchecked_slice_no_bounds_check_mut(s: &mut [u8]) -> u8 {
    let a = *s.get_unchecked_mut(2);
    // CHECK-NOT: panic_bounds_check
    a + s[1]
}

// CHECK-LABEL: @unchecked_slice_no_bounds_check_range
#[no_mangle]
pub unsafe fn unchecked_slice_no_bounds_check_range(s: &[u8]) -> u8 {
    let _a = &s.get_unchecked(..1);
    // CHECK-NOT: panic_bounds_check
    s[0]
}

// CHECK-LABEL: @unchecked_slice_no_bounds_check_range_mut
#[no_mangle]
pub unsafe fn unchecked_slice_no_bounds_check_range_mut(s: &mut [u8]) -> u8 {
    let _a = &mut s.get_unchecked_mut(..2);
    // CHECK-NOT: panic_bounds_check
    s[1]
}
