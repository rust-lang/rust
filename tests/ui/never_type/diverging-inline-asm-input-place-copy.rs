//@ build-pass
// Regression test for #154904
#![crate_type = "lib"]
#![feature(asm_unwind)] // for `bar`/`bar2`

#[inline(never)]
pub fn foo(x: &!) {
    unsafe {
        std::arch::asm!(
            "/* {} */",
            in(reg) *x,
        );
    }
}

#[inline(never)]
pub fn bar(x: &!) {
    unsafe {
        std::arch::asm!(
            "/* {} */",
            in(reg) *x,
            options(may_unwind),
        );
    }
}

// in case we make `&!` uninhabited in the future, also test the a copy from
// an unsafe place still doesn't trigger the ICE
#[inline(never)]
pub unsafe fn foo2(x: *const !) {
    unsafe {
        std::arch::asm!(
            "/* {} */",
            in(reg) *x,
        );
    }
}

#[inline(never)]
pub unsafe fn bar2(x: *const !) {
    unsafe {
        std::arch::asm!(
            "/* {} */",
            in(reg) *x,
            options(may_unwind),
        );
    }
}
