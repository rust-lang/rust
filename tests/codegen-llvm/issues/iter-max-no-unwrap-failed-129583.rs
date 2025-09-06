// Tests that `unwrap` is optimized out when the slice has a known length.
// The iterator may unroll for values smaller than a certain threshold so we
// use a larger value to prevent unrolling.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @infallible_max_not_unrolled
#[no_mangle]
pub fn infallible_max_not_unrolled(x: &[u8; 1024]) -> u8 {
    // CHECK-NOT: panic
    // CHECK-NOT: unwrap_failed
    *x.iter().max().unwrap()
}

// CHECK-LABEL: @infallible_max_unrolled
#[no_mangle]
pub fn infallible_max_unrolled(x: &[u8; 10]) -> u8 {
    // CHECK-NOT: panic
    // CHECK-NOT: unwrap_failed
    *x.iter().max().unwrap()
}

// CHECK-LABEL: @may_panic_max
#[no_mangle]
pub fn may_panic_max(x: &[u8]) -> u8 {
    // CHECK: unwrap_failed
    *x.iter().max().unwrap()
}
