// no-system-llvm
// compile-flags: -O
// ignore-debug: the debug assertions get in the way
#![crate_type = "lib"]

// Make sure no bounds checks are emitted in the loop when upfront slicing
// ensures that the slices are big enough.
// In particular, bounds checks were not always optimized out if the upfront
// check was for a greater len than the loop requires.
// (i.e. `already_sliced_no_bounds_check` was not always optimized even when
// `already_sliced_no_bounds_check_exact` was)
// CHECK-LABEL: @already_sliced_no_bounds_check
#[no_mangle]
pub fn already_sliced_no_bounds_check(a: &[u8], b: &[u8], c: &mut [u8]) {
    // CHECK: slice_end_index_len_fail
    // CHECK-NOT: panic_bounds_check
    let _ = (&a[..2048], &b[..2048], &mut c[..2048]);
    for i in 0..1024 {
        c[i] = a[i] ^ b[i];
    }
}

// CHECK-LABEL: @already_sliced_no_bounds_check_exact
#[no_mangle]
pub fn already_sliced_no_bounds_check_exact(a: &[u8], b: &[u8], c: &mut [u8]) {
    // CHECK: slice_end_index_len_fail
    // CHECK-NOT: panic_bounds_check
    let _ = (&a[..1024], &b[..1024], &mut c[..1024]);
    for i in 0..1024 {
        c[i] = a[i] ^ b[i];
    }
}

// Make sure we're checking for the right thing: there can be a panic if the slice is too small.
// CHECK-LABEL: @already_sliced_bounds_check
#[no_mangle]
pub fn already_sliced_bounds_check(a: &[u8], b: &[u8], c: &mut [u8]) {
    // CHECK: slice_end_index_len_fail
    // CHECK: panic_bounds_check
    let _ = (&a[..1023], &b[..2048], &mut c[..2048]);
    for i in 0..1024 {
        c[i] = a[i] ^ b[i];
    }
}
