// needs-asm-support

#![feature(naked_functions)]
#![crate_type = "lib"]

use std::arch::asm;

#[naked]
pub unsafe extern "C" fn inline_none() {
    asm!("", options(noreturn));
}

#[naked]
#[inline]
//~^ ERROR cannot use additional code generation attributes with `#[naked]`
pub unsafe extern "C" fn inline_hint() {
    asm!("", options(noreturn));
}

#[naked]
#[inline(always)]
//~^ ERROR cannot use additional code generation attributes with `#[naked]`
pub unsafe extern "C" fn inline_always() {
    asm!("", options(noreturn));
}

#[naked]
#[inline(never)]
//~^ ERROR cannot use additional code generation attributes with `#[naked]`
pub unsafe extern "C" fn inline_never() {
    asm!("", options(noreturn));
}
