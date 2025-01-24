//@ needs-asm-support
//@ only-x86_64

// tests that `va_start` is not injected into naked functions

#![crate_type = "lib"]
#![feature(c_variadic)]
#![feature(naked_functions)]
#![no_std]

#[naked]
pub unsafe extern "C" fn c_variadic(_: usize, _: ...) {
    // CHECK-NOT: va_start
    // CHECK-NOT: alloca
    core::arch::naked_asm! {
        "ret",
    }
}
