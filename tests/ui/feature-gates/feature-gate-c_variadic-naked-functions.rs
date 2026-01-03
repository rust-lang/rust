//@ add-minicore
//@ compile-flags: --target x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86
//@ ignore-backends: gcc

#![feature(no_core, lang_items, rustc_attrs)]
#![feature(c_variadic, abi_x86_interrupt, naked_functions_rustic_abi)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(C)]
#[lang = "va_list"]
pub struct VaList;

#[unsafe(naked)]
unsafe extern "sysv64" fn c_variadic_sysv64(_: ...) {
    //~^ ERROR Naked c-variadic `extern "sysv64"` functions are unstable
    naked_asm!("ret")
}

#[unsafe(naked)]
unsafe extern "C" fn c_variadic_c(_: ...) {
    naked_asm!("ret")
}
