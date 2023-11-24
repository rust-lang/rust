// compile-flags: -O
// only-x86_64
// ignore-debug: the debug assertions get in the way

#![crate_type = "lib"]

use std::alloc::Layout;

type RGB48 = [u16; 3];

// CHECK-LABEL: @layout_array_rgb48
#[no_mangle]
pub fn layout_array_rgb48(n: usize) -> Layout {
    // CHECK-NOT: icmp
    // CHECK-NOT: mul
    // CHECK: %[[TUP:.+]] = tail call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %n, i64 12)
    // CHECK-NOT: icmp
    // CHECK-NOT: mul
    // CHECK: %[[PROD:.+]] = extractvalue { i64, i1 } %[[TUP]], 0
    // CHECK-NEXT: lshr exact i64 %[[PROD]], 1
    // CHECK-NOT: icmp
    // CHECK-NOT: mul

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

// CHECK: declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
