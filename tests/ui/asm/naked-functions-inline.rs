//@ needs-asm-support
#![feature(naked_functions)]
#![crate_type = "lib"]

use std::arch::asm;

#[naked]
pub unsafe extern "C" fn inline_none() {
    asm!("", options(noreturn));
}

#[naked]
#[inline]
//~^ ERROR [E0736]
pub unsafe extern "C" fn inline_hint() {
    asm!("", options(noreturn));
}

#[naked]
#[inline(always)]
//~^ ERROR [E0736]
pub unsafe extern "C" fn inline_always() {
    asm!("", options(noreturn));
}

#[naked]
#[inline(never)]
//~^ ERROR [E0736]
pub unsafe extern "C" fn inline_never() {
    asm!("", options(noreturn));
}

#[naked]
#[cfg_attr(all(), inline(never))]
//~^ ERROR [E0736]
pub unsafe extern "C" fn conditional_inline_never() {
    asm!("", options(noreturn));
}
