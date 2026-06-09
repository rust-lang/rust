//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

/// Make sure no bounds checks are emitted after a `get_unchecked`.
// CHECK-LABEL: @unchecked_slice_no_bounds_check
#[no_mangle]
pub unsafe fn unchecked_slice_no_bounds_check(s: &[u8]) -> u8 {
    let a = *s.get_unchecked(1);
    // CHECK-NOT: panic_bounds_check
    a + s[0]
}
