//@ run-pass
//@ only-arm
//@ ignore-thumb (this test uses arm assembly)
//@ only-eabihf (the assembly below requires float hardware support)

// Check that multiple c-variadic calling conventions can be used in the same program.
//
// Clang and gcc reject defining functions with a non-default calling convention and a variable
// argument list, so C programs that use multiple c-variadic calling conventions are unlikely
// to come up. Here we validate that our codegen backends do in fact generate correct code.

extern "C" {
    fn variadic_c(_: f64, _: ...) -> f64;
}

extern "aapcs" {
    fn variadic_aapcs(_: f64, _: ...) -> f64;
}

fn main() {
    unsafe {
        assert_eq!(variadic_c(1.0, 2.0, 3.0), 1.0 + 2.0 + 3.0);
        assert_eq!(variadic_aapcs(1.0, 2.0, 3.0), 1.0 + 2.0 + 3.0);
    }
}

// This assembly was generated using https://godbolt.org/z/xcW6a1Tj5, and corresponds to the
// following code compiled for the `armv7-unknown-linux-gnueabihf` target:
//
// ```rust
// #![feature(c_variadic)]
//
// #[unsafe(no_mangle)]
// unsafe extern "C" fn variadic(a: f64, mut args: ...) -> f64 {
//     let b = args.arg::<f64>();
//     let c = args.arg::<f64>();
//
//     a + b + c
// }
// ```
//
// This function uses floats (and passes one normal float argument) because the aapcs and C calling
// conventions differ in how floats are passed, e.g. https://godbolt.org/z/sz799f51x. However, for
// c-variadic functions, both ABIs actually behave the same, based on:
//
// https://github.com/ARM-software/abi-aa/blob/main/aapcs32/aapcs32.rst#65parameter-passing
//
// > A variadic function is always marshaled as for the base standard.
//
// https://github.com/ARM-software/abi-aa/blob/main/aapcs32/aapcs32.rst#7the-standard-variants
//
// > This section applies only to non-variadic functions. For a variadic function the base standard
// > is always used both for argument passing and result return.
core::arch::global_asm!(
    r#"
{variadic_c}:
{variadic_aapcs}:
        sub     sp, sp, #12
        stmib   sp, {{r2, r3}}
        vmov    d0, r0, r1
        add     r0, sp, #4
        vldr    d1, [sp, #4]
        add     r0, r0, #15
        bic     r0, r0, #7
        vadd.f64        d0, d0, d1
        add     r1, r0, #8
        str     r1, [sp]
        vldr    d1, [r0]
        vadd.f64        d0, d0, d1
        vmov    r0, r1, d0
        add     sp, sp, #12
        bx      lr
    "#,
    variadic_c = sym variadic_c,
    variadic_aapcs = sym variadic_aapcs,
);
