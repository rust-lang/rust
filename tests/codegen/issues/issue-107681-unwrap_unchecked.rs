//@ compile-flags: -Copt-level=3

// Test for #107681.
// Make sure we don't create `br` or `select` instructions.

#![crate_type = "lib"]

use std::iter::Copied;
use std::slice::Iter;

#[no_mangle]
pub unsafe fn foo(x: &mut Copied<Iter<'_, u32>>) -> u32 {
    // CHECK-LABEL: @foo(
    // CHECK-NOT: br
    // CHECK-NOT: select
    // CHECK: [[RET:%.*]] = load i32, ptr
    // CHECK-NEXT: ret i32 [[RET]]
    x.next().unwrap_unchecked()
}
