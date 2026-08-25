// Tests that the slice access for `j` doesn't have a bounds check panic after
// being asserted as less than half of the slice length.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @check_only_assert_panic
#[no_mangle]
pub fn check_only_assert_panic(arr: &[u32], j: usize) -> u32 {
    // CHECK-NOT: panic_bounds_check
    assert!(j < arr.len() / 2);
    arr[j]
}
