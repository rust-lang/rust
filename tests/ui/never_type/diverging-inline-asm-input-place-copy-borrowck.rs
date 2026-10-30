//@ check-fail
//@ needs-asm-support
// Regression test for prior version of #161781
#![crate_type = "lib"]

#[inline(never)]
pub unsafe fn output(p: *const !, x: &usize) {
    unsafe {
        core::arch::asm!(
            //~^ ERROR: cannot assign to `*x`, which is behind a `&` reference
            "/* {} {} */",
            in(reg) *p,
            out(reg) *x,
        );
    }
}

#[inline(never)]
pub const unsafe fn const_asm(p: *const !) {
    // Normally asm is not allowed in const function, but this now bypasses the const check
    unsafe {
        core::arch::asm!("/* {} */", in(reg) *p);
        //~^ ERROR: inline assembly is not allowed in constant functions
    }
}
