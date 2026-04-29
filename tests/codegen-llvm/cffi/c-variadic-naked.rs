//@ needs-asm-support
//@ needs-asm-mnemonic: ret

// tests that `va_start` is not injected into naked functions

#![crate_type = "lib"]
#![feature(c_variadic)]
#![no_std]

#[unsafe(naked)]
pub unsafe extern "C" fn c_variadic(_: usize, _: ...) {
    // CHECK-NOT: va_start
    // CHECK-NOT: alloca
    core::arch::naked_asm!("ret")
}
