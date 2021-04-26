// needs-asm-support
// check-pass

#![crate_type = "rlib"]

use std::arch::asm;

pub unsafe fn asm1() {
    let _: () = asm!("");
}

pub unsafe fn asm2() {
    let _: ! = asm!("", options(noreturn));
}

pub unsafe fn asm3() -> ! {
    asm!("", options(noreturn));
}
