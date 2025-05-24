//@ should-fail
//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled
//! FIXME(#49892)
//! Tests that LLVM does not fully optimize comparisons of `Option<bool>`.
//! If this starts passing, it can be moved to `tests/codegen/option-niche-eq.rs`
#![crate_type = "lib"]

// CHECK-LABEL: @bool_eq
#[no_mangle]
pub fn bool_eq(l: Option<bool>, r: Option<bool>) -> bool {
    // CHECK: start:
    // CHECK-NEXT: icmp eq i8
    // CHECK-NEXT: ret i1
    l == r
}
