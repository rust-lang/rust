//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled

// Tests that `size_of_val_raw` on a slice with element size of one (*const [u8]) can be proven
// to have a size of <= isize::MAX by LLVM.
// The `mul nsw nuw` in size calculations was not sufficient when the element size was 1, so
// an `assume` is used in that case instead. Also an element size of 0 can always return 0.

#![feature(layout_for_ptr)]
#![crate_type = "lib"]

use std::mem::size_of_val_raw;

// CHECK-LABEL: elem_size_one
// CHECK: start
// CHECK-NEXT: ret i1 true
#[unsafe(no_mangle)]
pub fn elem_size_one(x: *const [u8]) -> bool {
    unsafe { size_of_val_raw(x) <= isize::MAX.cast_unsigned() }
}

// CHECK-LABEL: elem_size_two
// CHECK: start
// CHECK-NEXT: ret i1 true
#[unsafe(no_mangle)]
pub fn elem_size_two(x: *const [u16]) -> bool {
    unsafe { size_of_val_raw(x) <= isize::MAX.cast_unsigned() }
}

// CHECK-LABEL: elem_size_zero
// CHECK: start
// CHECK-NEXT: ret i1 true
#[unsafe(no_mangle)]
pub fn elem_size_zero(x: *const [()]) -> bool {
    unsafe { size_of_val_raw(x) == 0 }
}
