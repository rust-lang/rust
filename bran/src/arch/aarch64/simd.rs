use core::arch::asm;

/// Initialize SIMD/FP on AArch64.
///
/// Sets CPACR_EL1 to allow FP/SIMD instructions at EL1 and EL0.
pub fn init_cpu() {
    unsafe {
        // CPACR_EL1: Architectural Feature Access Control Register
        // Bits [21:20] control FPEN (Floating Point Enable)
        // 0b11 = Catch nothing, allow access to FP/SIMD at EL0 and EL1
        let mut cpacr: u64;
        asm!("mrs {}, cpacr_el1", out(reg) cpacr, options(nomem, nostack, preserves_flags));
        cpacr |= 0b11 << 20;
        asm!("msr cpacr_el1, {}", in(reg) cpacr, options(nomem, nostack, preserves_flags));

        // Ensure changes are visible
        asm!("isb", options(nomem, nostack, preserves_flags));
    }
}

/// AArch64 SIMD State Layout
///
/// 32 registers * 16 bytes (Q0-Q31) = 512 bytes
/// + 8 bytes FPSR
/// + 8 bytes FPCR
/// = 528 bytes total.
///
/// Align to 16 bytes.
pub const STATE_LAYOUT: (usize, usize) = (528, 16);

/// Save V0-V31 and FPSR/FPCR.
///
/// # Safety
/// dst must be 16-byte aligned and at least 528 bytes.
#[inline(never)] // Avoid reg alloc issues in the function itself
pub unsafe fn save(dst: *mut u8) {
    let regs = dst as *mut u128; // Treat as qword array for convenience
    // Store Q0-Q31 (V0-V31)
    // We use `stp` for pairs
    unsafe {
        asm!(
            "stp q0, q1, [{0}, #0]",
            "stp q2, q3, [{0}, #32]",
            "stp q4, q5, [{0}, #64]",
            "stp q6, q7, [{0}, #96]",
            "stp q8, q9, [{0}, #128]",
            "stp q10, q11, [{0}, #160]",
            "stp q12, q13, [{0}, #192]",
            "stp q14, q15, [{0}, #224]",
            "stp q16, q17, [{0}, #256]",
            "stp q18, q19, [{0}, #288]",
            "stp q20, q21, [{0}, #320]",
            "stp q22, q23, [{0}, #352]",
            "stp q24, q25, [{0}, #384]",
            "stp q26, q27, [{0}, #416]",
            "stp q28, q29, [{0}, #448]",
            "stp q30, q31, [{0}, #480]",
            in(reg) regs,
            options(nostack, preserves_flags)
        );
    }

    // Store FPSR and FPCR at offset 512
    unsafe {
        let status_ptr = dst.add(512) as *mut u64;
        let fpsr: u64;
        let fpcr: u64;
        asm!("mrs {}, fpsr", out(reg) fpsr, options(nomem, nostack, preserves_flags));
        asm!("mrs {}, fpcr", out(reg) fpcr, options(nomem, nostack, preserves_flags));

        *status_ptr = fpsr;
        *status_ptr.add(1) = fpcr;
    }
}

/// Restore V0-V31 and FPSR/FPCR.
///
/// # Safety
/// src must be valid and contain saved state.
#[inline(never)]
pub unsafe fn restore(src: *const u8) {
    let regs = src as *const u128;

    // Restore Q0-Q31
    unsafe {
        asm!(
            "ldp q0, q1, [{0}, #0]",
            "ldp q2, q3, [{0}, #32]",
            "ldp q4, q5, [{0}, #64]",
            "ldp q6, q7, [{0}, #96]",
            "ldp q8, q9, [{0}, #128]",
            "ldp q10, q11, [{0}, #160]",
            "ldp q12, q13, [{0}, #192]",
            "ldp q14, q15, [{0}, #224]",
            "ldp q16, q17, [{0}, #256]",
            "ldp q18, q19, [{0}, #288]",
            "ldp q20, q21, [{0}, #320]",
            "ldp q22, q23, [{0}, #352]",
            "ldp q24, q25, [{0}, #384]",
            "ldp q26, q27, [{0}, #416]",
            "ldp q28, q29, [{0}, #448]",
            "ldp q30, q31, [{0}, #480]",
            in(reg) regs,
            options(nostack, preserves_flags)
        );
    }

    // Restore FPSR/FPCR
    unsafe {
        let status_ptr = src.add(512) as *const u64;
        let fpsr = *status_ptr;
        let fpcr = *status_ptr.add(1);

        asm!("msr fpsr, {}", in(reg) fpsr, options(nomem, nostack, preserves_flags));
        asm!("msr fpcr, {}", in(reg) fpcr, options(nomem, nostack, preserves_flags));
    }
}
