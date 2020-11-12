// compile-flags: -O -Zmerge-functions=disabled
// ignore-x86
// ignore-arm
// ignore-emscripten
// ignore-gnux32
// ignore 32-bit platforms (LLVM has a bug with them)

// Check that LLVM understands that `Iter` pointer is not null. Issue #37945.

#![crate_type = "lib"]

use std::slice::Iter;

#[no_mangle]
pub fn is_empty_1(xs: Iter<f32>) -> bool {
// CHECK-LABEL: @is_empty_1(
// CHECK-NEXT:  start:
// CHECK-NEXT:    [[A:%.*]] = icmp ne i32* %xs.1, null
// CHECK-NEXT:    tail call void @llvm.assume(i1 [[A]])
// CHECK-NEXT:    [[B:%.*]] = icmp eq i32* %xs.0, %xs.1
// CHECK-NEXT:    ret i1 [[B:%.*]]
    {xs}.next().is_none()
}

#[no_mangle]
pub fn is_empty_2(xs: Iter<f32>) -> bool {
// CHECK-LABEL: @is_empty_2
// CHECK-NEXT:  start:
// CHECK-NEXT:    [[C:%.*]] = icmp ne i32* %xs.1, null
// CHECK-NEXT:    tail call void @llvm.assume(i1 [[C]])
// CHECK-NEXT:    [[D:%.*]] = icmp eq i32* %xs.0, %xs.1
// CHECK-NEXT:    ret i1 [[D:%.*]]
    xs.map(|&x| x).next().is_none()
}
