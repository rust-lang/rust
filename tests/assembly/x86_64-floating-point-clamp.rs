// Floating-point clamp is designed to be implementable as max+min,
// so check to make sure that's what it's actually emitting.

//@ assembly-output: emit-asm
// Set the base cpu explicitly, in case the default has been changed.
//@ compile-flags: --crate-type=lib -Copt-level=3 -C llvm-args=-x86-asm-syntax=intel -C target-cpu=x86-64
//@ only-x86_64
//@ ignore-sgx

// CHECK-LABEL: clamp_demo:
#[no_mangle]
pub fn clamp_demo(a: f32, x: f32, y: f32) -> f32 {
    // CHECK: maxss
    // CHECK: minss
    a.clamp(x, y)
}

// CHECK-LABEL: clamp12_demo:
#[no_mangle]
pub fn clamp12_demo(a: f32) -> f32 {
    // CHECK: movss   xmm1
    // CHECK-NEXT: maxss   xmm1, xmm0
    // CHECK-NEXT: movss   xmm0
    // CHECK-NEXT: minss   xmm0, xmm1
    // CHECK: ret
    a.clamp(1.0, 2.0)
}
