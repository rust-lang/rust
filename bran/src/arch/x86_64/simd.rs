use core::arch::asm;

/// Initialize SIMD (SSE/SSE2) on this CPU.
///
/// This enables SSE by setting CR0.MP, CR4.OSFXSR, and CR4.OSXMMEXCPT,
/// and clearing CR0.EM.
pub fn init_cpu() {
    // CR0: Clear EM (bit 2), Set MP (bit 1)
    // EM = Emulation (if set, FPU instructions trap) - we want it clear
    // MP = Monitor Co-processor (controls WAIT/FWAIT interaction with TS flag)
    let mut cr0: u64;
    unsafe { asm!("mov {}, cr0", out(reg) cr0, options(nomem, nostack, preserves_flags)) };
    cr0 &= !(1 << 2);
    cr0 |= 1 << 1;
    unsafe { asm!("mov cr0, {}", in(reg) cr0, options(nomem, nostack, preserves_flags)) };

    // CR4: Set OSFXSR (bit 9) and OSXMMEXCPT (bit 10)
    // OSFXSR = Enable SSE support (FXSAVE/FXRSTOR)
    // OSXMMEXCPT = Enable unmasked SSE exceptions
    let mut cr4: u64;
    unsafe { asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags)) };
    cr4 |= (1 << 9) | (1 << 10);
    unsafe { asm!("mov cr4, {}", in(reg) cr4, options(nomem, nostack, preserves_flags)) };
}

/// FXSAVE area size is 512 bytes, 16-byte aligned.
pub const STATE_LAYOUT: (usize, usize) = (512, 16);

/// Save SSE state to 512-byte buffer using FXSAVE.
///
/// # Safety
/// dst must be 16-byte aligned and point to 512 bytes of writable memory.
#[inline]
pub unsafe fn save(dst: *mut u8) {
    unsafe { asm!("fxsave [{}]", in(reg) dst, options(nomem, nostack, preserves_flags)) };
}

/// Restore SSE state from 512-byte buffer using FXRSTOR.
///
/// # Safety
/// src must be 16-byte aligned and point to valid FXSAVE data.
#[inline]
pub unsafe fn restore(src: *const u8) {
    unsafe { asm!("fxrstor [{}]", in(reg) src, options(nomem, nostack, preserves_flags)) };
}
