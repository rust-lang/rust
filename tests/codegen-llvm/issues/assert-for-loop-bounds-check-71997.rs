// Tests that there's no bounds check within for-loop after asserting that
// the range start and end are within bounds.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @no_bounds_check_after_assert
#[no_mangle]
fn no_bounds_check_after_assert(slice: &[u64], start: usize, end: usize) -> u64 {
    // CHECK-NOT: panic_bounds_check
    let mut total = 0;
    assert!(start < end && start < slice.len() && end <= slice.len());
    for i in start..end {
        total += slice[i];
    }
    total
}
