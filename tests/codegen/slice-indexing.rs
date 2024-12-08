//@ compile-flags: -O
//@ only-64bit (because the LLVM type of i64 for usize shows up)

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

// CHECK-LABEL: @index_mut_by_range(
#[no_mangle]
pub fn index_mut_by_range(x: &mut [i32], r: Range<usize>) -> &mut [i32] {
    // CHECK: sub nuw i64
    &mut x[r]
}

// CHECK-LABEL: @get_unchecked_mut_by_range(
#[no_mangle]
pub unsafe fn get_unchecked_mut_by_range(x: &mut [i32], r: Range<usize>) -> &mut [i32] {
    // CHECK: sub nuw i64
    x.get_unchecked_mut(r)
}

// CHECK-LABEL: @str_index_by_range(
#[no_mangle]
pub fn str_index_by_range(x: &str, r: Range<usize>) -> &str {
    // CHECK: sub nuw i64
    &x[r]
}

// CHECK-LABEL: @str_get_unchecked_by_range(
#[no_mangle]
pub unsafe fn str_get_unchecked_by_range(x: &str, r: Range<usize>) -> &str {
    // CHECK: sub nuw i64
    x.get_unchecked(r)
}

// CHECK-LABEL: @str_index_mut_by_range(
#[no_mangle]
pub fn str_index_mut_by_range(x: &mut str, r: Range<usize>) -> &mut str {
    // CHECK: sub nuw i64
    &mut x[r]
}

// CHECK-LABEL: @str_get_unchecked_mut_by_range(
#[no_mangle]
pub unsafe fn str_get_unchecked_mut_by_range(x: &mut str, r: Range<usize>) -> &mut str {
    // CHECK: sub nuw i64
    x.get_unchecked_mut(r)
}
