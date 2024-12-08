//@ assembly-output: emit-asm
//@ compile-flags: -O
//@ only-windows
//@ only-x86_64

#![feature(f16, f128)]
#![crate_type = "lib"]

// CHECK-LABEL: second_f16
// CHECK: movaps %xmm1, %xmm0
// CHECK-NEXT: retq
#[no_mangle]
pub extern "C" fn second_f16(_: f16, x: f16) -> f16 {
    x
}

// CHECK-LABEL: second_f32
// CHECK: movaps %xmm1, %xmm0
// CHECK-NEXT: retq
#[no_mangle]
pub extern "C" fn second_f32(_: f32, x: f32) -> f32 {
    x
}

// CHECK-LABEL: second_f64
// CHECK: movaps %xmm1, %xmm0
// CHECK-NEXT: retq
#[no_mangle]
pub extern "C" fn second_f64(_: f64, x: f64) -> f64 {
    x
}

// CHECK-LABEL: second_f128
// CHECK: movaps %xmm1, %xmm0
// CHECK-NEXT: retq
#[no_mangle]
pub extern "C" fn second_f128(_: f128, x: f128) -> f128 {
    x
}
