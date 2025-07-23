//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

use std::iter::StepBy;
use std::slice::Iter;

// The constructor for `StepBy` ensures we can never end up needing to do zero
// checks on denominators, so check that the code isn't emitting panic paths.

// CHECK-LABEL: @step_by_len_std
#[no_mangle]
pub fn step_by_len_std(x: &StepBy<Iter<i32>>) -> usize {
    // CHECK-NOT: div_by_zero
    // CHECK: udiv
    // CHECK-NOT: div_by_zero
    x.len()
}

// CHECK-LABEL: @step_by_len_naive
#[no_mangle]
pub fn step_by_len_naive(x: Iter<i32>, step_minus_one: usize) -> usize {
    // CHECK: udiv
    // CHECK: call{{.+}}div_by_zero
    x.len() / (step_minus_one + 1)
}
