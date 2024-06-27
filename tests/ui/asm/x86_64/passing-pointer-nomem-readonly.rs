//@ only-x86_64
//@ needs-asm-support
//@ build-pass

#![crate_type = "lib"]
#![no_std]

unsafe fn nomem_bad(p: &i32) {
    core::arch::asm!("mov {p}, {p}", p = in(reg) p, options(nomem, nostack, preserves_flags));
    //~^ WARNING passing a pointer to asm! block with 'nomem' option.
}

unsafe fn readonly_bad(p: &mut i32) {
    core::arch::asm!("mov {p}, {p}", p = in(reg) p, options(readonly, nostack, preserves_flags));
    //~^ WARNING passing a mutable pointer to asm! block with 'readonly' option.
}

unsafe fn nomem_good(p: &i32) {
    core::arch::asm!("mov {p}, {p}", p = in(reg) p, options(readonly, nostack, preserves_flags));
    let p = p as *const i32 as usize;
    core::arch::asm!("mov {p}, {p}", p = in(reg) p, options(nomem, nostack, preserves_flags));
}

unsafe fn readonly_good(p: &mut i32) {
    core::arch::asm!("mov {p}, {p}", p = in(reg) p, options(nostack, preserves_flags));
    core::arch::asm!("mov {p}, {p}", p = in(reg) &*p, options(readonly, nostack, preserves_flags));
    let p = p as *const i32;
    core::arch::asm!("mov {p}, {p}", p = in(reg) p, options(readonly, nostack, preserves_flags));
}

unsafe fn nomem_bad2(p: &mut i32) {
    core::arch::asm!("mov {p}, {p}", p = in(reg) p, options(nomem, nostack, preserves_flags));
    //~^ WARNING passing a pointer to asm! block with 'nomem' option.
}
