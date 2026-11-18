//! Regression test for <https://github.com/rust-lang/rust/issues/57517>.
//!
//! A loop over `iter_mut().step_by(1)` must be vectorized, like the same loop
//! without `step_by`.

//@ compile-flags: -Copt-level=3
//@ only-x86_64

#![crate_type = "lib"]

// CHECK-LABEL: @step_by_one
#[no_mangle]
pub fn step_by_one(arr: &mut [i32]) {
    // CHECK: add <{{[0-9]+}} x i32>
    for e in arr.iter_mut().step_by(1) {
        *e = e.wrapping_add(3);
    }
}
