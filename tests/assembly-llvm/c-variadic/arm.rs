//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3
//@ only-arm
//@ ignore-thumb
//@ ignore-android
#![no_std]
#![crate_type = "lib"]

// Check that the assembly that rustc generates matches what clang emits. This example in particular
// is related to https://github.com/rust-lang/rust/pull/144549 and shows the effect of us correctly
// emitting annotations that start and end the lifetime of the va_list.

#[unsafe(no_mangle)]
unsafe extern "C" fn variadic(a: f64, mut args: ...) -> f64 {
    // CHECK-LABEL: variadic
    // CHECK: sub sp, sp

    // CHECK: vldr
    // CHECK: vadd.f64
    // CHECK: vldr
    // CHECK: vadd.f64
    let b = args.next_arg::<f64>();
    let c = args.next_arg::<f64>();
    a + b + c

    // CHECK: add sp, sp
}
