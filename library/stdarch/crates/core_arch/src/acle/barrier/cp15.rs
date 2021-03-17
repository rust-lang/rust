// Reference: ARM11 MPCore Processor Technical Reference Manual (ARM DDI 0360E) Section 3.5 "Summary
// of CP15 instructions"

/// Full system is the required shareability domain, reads and writes are the
/// required access types
pub struct SY;

impl super::super::sealed::Dmb for SY {
    #[inline(always)]
    unsafe fn __dmb(&self) {
        asm!(
            "mcr p15, 0, r0, c7, c10, 5",
            options(preserves_flags, nostack)
        )
    }
}

impl super::super::sealed::Dsb for SY {
    #[inline(always)]
    unsafe fn __dsb(&self) {
        asm!(
            "mcr p15, 0, r0, c7, c10, 4",
            options(preserves_flags, nostack)
        )
    }
}

impl super::super::sealed::Isb for SY {
    #[inline(always)]
    unsafe fn __isb(&self) {
        asm!(
            "mcr p15, 0, r0, c7, c5, 4",
            options(preserves_flags, nostack)
        )
    }
}
