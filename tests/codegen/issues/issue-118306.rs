//@ compile-flags: -O -Zno-jump-tables
//@ min-llvm-version: 19

// Test for #118306.
// Ensure that the default branch is optimized to be unreachable.

#![crate_type = "lib"]

#[no_mangle]
pub fn foo(input: u64) -> u64 {
    // CHECK-LABEL: @foo(
    // CHECK: switch {{.*}}, label %[[UNREACHABLE:.*]] [
    // CHECK: [[UNREACHABLE]]:
    // CHECK-NEXT: unreachable
    match input % 4 {
        1 | 2 => 1,
        3 => 2,
        _ => 0,
    }
}
