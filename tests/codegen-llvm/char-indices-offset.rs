//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]
#![no_std]

use core::convert::TryFrom;

// CHECK-LABEL: @char_indices_offset_fits_isize
// CHECK: ret i1 true
#[no_mangle]
pub fn char_indices_offset_fits_isize(iter: &core::str::CharIndices<'_>) -> bool {
    iter.offset() <= isize::MAX as usize
}

// CHECK-LABEL: @char_indices_offset_as_isize
// CHECK-NOT: panic
// CHECK: load i{{32|64}}
// CHECK: llvm.assume
// CHECK-NOT: panic
// CHECK: ret i{{32|64}}
#[no_mangle]
pub fn char_indices_offset_as_isize(iter: &core::str::CharIndices<'_>) -> isize {
    isize::try_from(iter.offset()).unwrap()
}
