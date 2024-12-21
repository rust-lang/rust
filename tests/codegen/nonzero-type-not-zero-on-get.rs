//@ compile-flags: -C opt-level=3
//! Ensure that `.get()` on `std::num::NonZero*` types do not
//! check for zero equivalency.
//! Discovered in issue #49572.

#![crate_type = "lib"]

#[no_mangle]
pub fn foo(x: std::num::NonZeroU32) -> bool {
    // CHECK-LABEL: @foo(
    // CHECK: ret i1 true
    x.get() != 0
}

#[no_mangle]
pub fn bar(x: std::num::NonZeroI64) -> bool {
    // CHECK-LABEL: @bar(
    // CHECK: ret i1 true
    x.get() != 0
}
