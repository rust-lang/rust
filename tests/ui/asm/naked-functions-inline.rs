//@ needs-asm-support
#![crate_type = "lib"]

use std::arch::naked_asm;

#[unsafe(naked)]
pub extern "C" fn inline_none() {
    naked_asm!("");
}

#[unsafe(naked)]
#[inline]
//~^ ERROR [E0736]
pub extern "C" fn inline_hint() {
    naked_asm!("");
}

#[unsafe(naked)]
#[inline(always)]
//~^ ERROR [E0736]
pub extern "C" fn inline_always() {
    naked_asm!("");
}

#[unsafe(naked)]
#[inline(never)]
//~^ ERROR [E0736]
pub extern "C" fn inline_never() {
    naked_asm!("");
}

#[unsafe(naked)]
#[cfg_attr(all(), inline(never))]
//~^ ERROR [E0736]
pub extern "C" fn conditional_inline_never() {
    naked_asm!("");
}
