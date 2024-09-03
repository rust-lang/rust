//@ needs-asm-support
#![warn(clippy::pointers_in_nomem_asm_block)]
#![crate_type = "lib"]
#![no_std]

use core::arch::asm;

unsafe fn nomem_bad(p: &i32) {
    asm!(
        "asdf {p1}, {p2}, {p3}",
        p1 = in(reg) p,
        //~^ ERROR: passing pointers to nomem asm block
        p2 = in(reg) p as *const _ as usize,
        p3 = in(reg) p,
        options(nomem, nostack, preserves_flags)
    );
}

unsafe fn nomem_good(p: &i32) {
    asm!("asdf {p}", p = in(reg) p, options(readonly, nostack, preserves_flags));
    let p = p as *const i32 as usize;
    asm!("asdf {p}", p = in(reg) p, options(nomem, nostack, preserves_flags));
}

unsafe fn nomem_bad2(p: &mut i32) {
    asm!("asdf {p}", p = in(reg) p, options(nomem, nostack, preserves_flags));
    //~^ ERROR: passing pointers to nomem asm block
}

unsafe fn nomem_fn(p: extern "C" fn()) {
    asm!("call {p}", p = in(reg) p, options(nomem));
    //~^ ERROR: passing pointers to nomem asm block
}
