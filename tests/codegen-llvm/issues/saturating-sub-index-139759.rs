// Test that calculating an index with saturating subtraction from an in-bounds
// index doesn't generate another bounds check.

//@ compile-flags: -Copt-level=3
//@ min-llvm-version: 21

#![crate_type = "lib"]

// CHECK-LABEL: @bounds_check_is_elided
#[no_mangle]
pub fn bounds_check_is_elided(s: &[i32], index: usize) -> i32 {
    // CHECK-NOT: panic_bounds_check
    if index < s.len() {
        let lower_bound = index.saturating_sub(1);
        s[lower_bound]
    } else {
        -1
    }
}
