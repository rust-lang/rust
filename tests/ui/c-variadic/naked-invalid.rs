//@ add-minicore
//@ compile-flags: --target x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86
//@ ignore-backends: gcc

#![feature(no_core, lang_items, rustc_attrs)]
#![feature(c_variadic, c_variadic_naked_functions, abi_x86_interrupt, naked_functions_rustic_abi)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(C)]
#[lang = "va_list"]
pub struct VaList;

#[unsafe(naked)]
unsafe extern "sysv64" fn c_variadic_sysv64(_: ...) {
    naked_asm!("ret")
}

#[unsafe(naked)]
unsafe extern "C" fn c_variadic_c(_: ...) {
    naked_asm!("ret")
}

#[unsafe(naked)]
unsafe extern "Rust" fn c_variadic_rust(_: ...) {
    //~^ ERROR `...` is not supported for `extern "Rust"` naked functions
    naked_asm!("ret")
}

#[unsafe(naked)]
unsafe extern "x86-interrupt" fn c_variadic_x86_interrupt(_: ...) {
    //~^ ERROR `...` is not supported for `extern "x86-interrupt"` naked functions
    naked_asm!("ret")
}

#[unsafe(naked)]
unsafe extern "nonsense" fn c_variadic_x86_nonsense(_: ...) {
    //~^ ERROR invalid ABI: found `nonsense`
    naked_asm!("ret")
}
