// no-system-llvm
// compile-flags: -O
#![crate_type = "lib"]

// CHECK-LABEL: @already_sliced_no_bounds_check
#[no_mangle]
pub fn already_sliced_no_bounds_check(a: &[u8], b: &[u8], c: &mut [u8]) {
    // CHECK: slice_index_len_fail
    // CHECK-NOT: panic_bounds_check
    let _ = (&a[..2048], &b[..2048], &mut c[..2048]);
    for i in 0..1024 {
        c[i] = a[i] ^ b[i];
    }
}

// make sure we're checking for the right thing: there can be a panic if the slice is too small
// CHECK-LABEL: @already_sliced_bounds_check
#[no_mangle]
pub fn already_sliced_bounds_check(a: &[u8], b: &[u8], c: &mut [u8]) {
    // CHECK: slice_index_len_fail
    // CHECK: panic_bounds_check
    let _ = (&a[..1023], &b[..2048], &mut c[..2048]);
    for i in 0..1024 {
        c[i] = a[i] ^ b[i];
    }
}
