// compile-flags: -O
// ignore-x86
// ignore-arm
// ignore-emscripten
// ignore-gnux32
// ignore 32-bit platforms (LLVM has a bug with them)

// See issue #37945.

#![crate_type = "lib"]

use std::slice::Iter;

// CHECK-LABEL: @is_empty_1
#[no_mangle]
pub fn is_empty_1(xs: Iter<f32>) -> bool {
// CHECK-NOT: icmp eq float* {{.*}}, null
    {xs}.next().is_none()
}

// CHECK-LABEL: @is_empty_2
#[no_mangle]
pub fn is_empty_2(xs: Iter<f32>) -> bool {
// CHECK-NOT: icmp eq float* {{.*}}, null
    xs.map(|&x| x).next().is_none()
}
