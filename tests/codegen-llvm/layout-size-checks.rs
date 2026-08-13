//@ compile-flags: -Copt-level=3
//@ only-x86_64

#![crate_type = "lib"]

use std::alloc::Layout;
use std::convert::TryFrom;

type RGB48 = [u16; 3];

// CHECK-LABEL: @layout_array_rgb48
#[no_mangle]
pub fn layout_array_rgb48(n: usize) -> Layout {
    // CHECK-NOT: llvm.umul.with.overflow.i64
    // CHECK: icmp ugt i64 %n, 1537228672809129301
    // CHECK-NOT: llvm.umul.with.overflow.i64
    // CHECK: mul nuw nsw i64 %n, 6
    // CHECK-NOT: llvm.umul.with.overflow.i64
    Layout::array::<RGB48>(n).unwrap()
}

// CHECK-LABEL: @layout_array_i32
#[no_mangle]
pub fn layout_array_i32(n: usize) -> Layout {
    // CHECK-NOT: llvm.umul.with.overflow.i64
    // CHECK: icmp ugt i64 %n, 2305843009213693951
    // CHECK-NOT: llvm.umul.with.overflow.i64
    // CHECK: shl nuw nsw i64 %n, 2
    // CHECK-NOT: llvm.umul.with.overflow.i64
    Layout::array::<i32>(n).unwrap()
}

// CHECK-LABEL: @layout_size_fits_in_isize
#[no_mangle]
pub fn layout_size_fits_in_isize(layout: &Layout) -> bool {
    // CHECK: ret i1 true
    layout.size() <= isize::MAX as usize
}

// CHECK-LABEL: @layout_size_to_isize
#[no_mangle]
pub fn layout_size_to_isize(layout: &Layout) -> isize {
    // CHECK-NOT: panic
    // CHECK: load i64
    // CHECK-NOT: panic
    // CHECK: ret i64
    isize::try_from(layout.size()).unwrap()
}

// CHECK-LABEL: @layout_size_checked_double
#[no_mangle]
pub fn layout_size_checked_double(layout: &Layout) -> usize {
    // CHECK-NOT: panic
    // CHECK: shl nuw i64
    // CHECK-NOT: panic
    // CHECK: ret i64
    layout.size().checked_mul(2).unwrap()
}
