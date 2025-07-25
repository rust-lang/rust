//@ should-fail
//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled
//! FIXME(#49892)
//! Test that the derived implementation of `PartialEq` for `Option` is not fully
//! optimized by LLVM. If this starts passing, the test and manual impl should
//! be removed.
#![crate_type = "lib"]

use std::num::NonZero;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Option<T> {
    None,
    Some(T),
}

// CHECK-LABEL: @non_zero_eq
#[no_mangle]
pub fn non_zero_eq(l: Option<NonZero<u32>>, r: Option<NonZero<u32>>) -> bool {
    // CHECK: start:
    // CHECK-NEXT: icmp eq i32
    // CHECK-NEXT: ret i1
    l == r
}
