//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3
//@ only-arm
//@ ignore-thumb
#![no_std]
#![crate_type = "lib"]
#![feature(c_variadic)]

// Check that the assembly that rustc generates matches what clang emits.

#[unsafe(no_mangle)]
unsafe extern "C" fn variadic(a: f64, mut args: ...) -> f64 {
    // CHECK-LABEL: variadic
    // CHECK: sub sp, sp, #12

    // CHECK: vldr
    let b = args.arg::<f64>();
    // CHECK: vldr
    let c = args.arg::<f64>();

    // CHECK: vadd.f64
    // CHECK: vadd.f64
    a + b + c

    // CHECK: add sp, sp, #12
}
