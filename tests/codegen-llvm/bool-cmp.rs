// This is a test for optimal Ord trait implementation for bool.
// See <https://github.com/rust-lang/rust/issues/66780> for more info.

//@ compile-flags: -C opt-level=3

#![crate_type = "lib"]

use std::cmp::Ordering;

// CHECK-LABEL: @cmp_bool
#[no_mangle]
pub fn cmp_bool(a: bool, b: bool) -> Ordering {
    // LLVM 10 produces (zext a) + (sext b), but the final lowering is (zext a) - (zext b).
    // CHECK: zext i1
    // CHECK: {{z|s}}ext i1
    // CHECK: {{sub|add}} nsw
    a.cmp(&b)
}
