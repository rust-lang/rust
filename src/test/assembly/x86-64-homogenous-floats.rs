// assembly-output: emit-asm
// needs-llvm-components: x86
// compile-flags: --target x86_64-unknown-linux-gnu
// compile-flags: -C llvm-args=--x86-asm-syntax=intel
// compile-flags: -C opt-level=3

#![crate_type = "rlib"]
#![no_std]

// CHECK-LABEL: sum_f32:
// CHECK:      addss xmm0, xmm1
// CHECK-NEXT: ret
#[no_mangle]
pub fn sum_f32(a: f32, b: f32) -> f32 {
    a + b
}

// CHECK-LABEL: sum_f32x2:
// CHECK:      addss xmm{{[0-9]}}, xmm{{[0-9]}}
// CHECK-NEXT: addss xmm{{[0-9]}}, xmm{{[0-9]}}
// CHECK-NEXT: ret
#[no_mangle]
pub fn sum_f32x2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [
        a[0] + b[0],
        a[1] + b[1],
    ]
}

// CHECK-LABEL: sum_f32x4:
// CHECK:      mov     rax, [[PTR_IN:.*]]
// CHECK-NEXT: movups  [[XMMA:xmm[0-9]]], xmmword ptr [rsi]
// CHECK-NEXT: movups  [[XMMB:xmm[0-9]]], xmmword ptr [rdx]
// CHECK-NEXT: addps   [[XMMB]], [[XMMA]]
// CHECK-NEXT: movups  xmmword ptr {{\[}}[[PTR_IN]]{{\]}}, [[XMMB]]
// CHECK-NEXT: ret
#[no_mangle]
pub fn sum_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
    ]
}
