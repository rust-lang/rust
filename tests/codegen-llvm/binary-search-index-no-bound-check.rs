//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

// Make sure no bounds checks are emitted when slicing or indexing
// with an index from `binary_search`.

// CHECK-LABEL: @binary_search_index_no_bounds_check
#[no_mangle]
pub fn binary_search_index_no_bounds_check(s: &[u8]) -> u8 {
    // CHECK-NOT: panic
    // CHECK-NOT: slice_index_fail
    // CHECK-NOT: panic_bounds_check
    if let Ok(idx) = s.binary_search(&b'\\') { s[idx] } else { 42 }
}

// Similarly, check that `partition_point` is known to return a valid fencepost.

// CHECK-LABEL: @unknown_split
#[no_mangle]
pub fn unknown_split(x: &[i32], i: usize) -> (&[i32], &[i32]) {
    // This just makes sure that the subsequent function is looking for the
    // absence of something that might actually be there.

    // CHECK: call core::panicking::panic
    x.split_at(i)
}

// CHECK-LABEL: @partition_point_split_no_bounds_check
#[no_mangle]
pub fn partition_point_split_no_bounds_check(x: &[i32], needle: i32) -> (&[i32], &[i32]) {
    // CHECK-NOT: call core::panicking::panic
    let i = x.partition_point(|p| p < &needle);
    x.split_at(i)
}
