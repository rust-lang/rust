//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

// This test verifies that LLVM 20 properly optimizes the bounds check
// when accessing the last few elements of a slice with proper conditions.
// Previously, this would generate an unreachable branch to
// slice_index_fail even when the bounds check was provably safe.

// CHECK-LABEL: @last_four_initial(
#[no_mangle]
pub fn last_four_initial(s: &[u8]) -> &[u8] {
    // Previously this would generate a branch to slice_index_fail
    // that is unreachable. The LLVM 20 fix should eliminate this branch.
    // CHECK-NOT: slice_index_fail
    // CHECK-NOT: unreachable
    let start = if s.len() <= 4 { 0 } else { s.len() - 4 };
    &s[start..]
}

// CHECK-LABEL: @last_four_optimized(
#[no_mangle]
pub fn last_four_optimized(s: &[u8]) -> &[u8] {
    // This version was already correctly optimized before the fix in LLVM 20.
    // CHECK-NOT: slice_index_fail
    // CHECK-NOT: unreachable
    if s.len() <= 4 { &s[0..] } else { &s[s.len() - 4..] }
}

// Just to verify we're correctly checking for the right thing
// CHECK-LABEL: @test_bounds_check_happens(
#[no_mangle]
pub fn test_bounds_check_happens(s: &[u8], i: usize) -> &[u8] {
    // CHECK: slice_index_fail
    &s[i..]
}
