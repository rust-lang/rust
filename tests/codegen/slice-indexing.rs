// compile-flags: -O
// only-64bit (because the LLVM type of i64 for usize shows up)
// ignore-debug: the debug assertions get in the way

#![crate_type = "lib"]

use std::ops::Range;

// CHECK-LABEL: @index_by_range(
#[no_mangle]
pub fn index_by_range(x: &[u16], r: Range<usize>) -> &[u16] {
    // CHECK: sub nuw i64
    &x[r]
}

// CHECK-LABEL: @get_unchecked_by_range(
#[no_mangle]
pub unsafe fn get_unchecked_by_range(x: &[u16], r: Range<usize>) -> &[u16] {
    // CHECK: sub nuw i64
    x.get_unchecked(r)
}
