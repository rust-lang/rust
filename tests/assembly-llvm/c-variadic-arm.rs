//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3
//@ only-arm
//@ ignore-thumb
//@ ignore-android
#![no_std]
#![crate_type = "lib"]
#![feature(c_variadic)]

// Check that the assembly that rustc generates matches what clang emits.

#[unsafe(no_mangle)]
unsafe extern "C" fn variadic(a: f64, mut args: ...) -> f64 {
    // CHECK-LABEL: variadic
    // CHECK: sub sp, sp

    // CHECK: vldr
    // CHECK: vadd.f64
    // CHECK: vldr
    // CHECK: vadd.f64
    let b = args.arg::<f64>();
    let c = args.arg::<f64>();
    a + b + c

    // CHECK: add sp, sp
}
