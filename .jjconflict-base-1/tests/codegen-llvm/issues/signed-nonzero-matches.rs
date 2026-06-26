// Matching a signed non-zero value against -1 or 1 should not require
// branching code.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

use std::num::NonZeroI32;

// CHECK-LABEL: @signed_nonzero_matches
#[no_mangle]
pub fn signed_nonzero_matches(n: NonZeroI32) -> bool {
    // CHECK-NOT: br i1
    // CHECK: ret i1
    matches!(n.get(), -1 | 1)
}

// CHECK-LABEL: @signed_nonzero_eq
#[no_mangle]
pub fn signed_nonzero_eq(n: NonZeroI32) -> bool {
    // CHECK-NOT: br i1
    // CHECK: ret i1
    n.get() == -1 || n.get() == 1
}
