//! x86_64 user-mode Thread-Local Storage base helpers.
//!
//! On x86_64, user-mode TLS is addressed through FS_BASE (MSR 0xC000_0100).
//! The kernel tracks the per-thread FS_BASE and saves/restores it on every
//! context switch so that each thread has isolated TLS state.
//!
//! # Kernel vs. user TLS separation
//!
//! The kernel uses GS_BASE for its own per-CPU state (CpuLocal) and relies on
//! `swapgs` at syscall entry/exit to switch between the kernel GS and the
//! user-saved GS.  FS_BASE is *not* used by the kernel and is therefore safe
//! to hand over completely to user-mode threads.

use core::arch::asm;

/// MSR address for FS.Base (user-mode TLS pointer).
const IA32_FS_BASE: u32 = 0xC000_0100;

/// Read the current user FS_BASE from the corresponding MSR.
///
/// # Safety
/// Requires RDMSR privilege (CPL 0).  Must not be called from user mode.
#[inline(always)]
pub fn read_user_fs_base() -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        asm!(
            "rdmsr",
            in("ecx") IA32_FS_BASE,
            out("eax") lo,
            out("edx") hi,
            options(nostack, preserves_flags, nomem),
        );
    }
    ((hi as u64) << 32) | (lo as u64)
}

/// Write a new value to the FS_BASE MSR.
///
/// # Safety
/// Requires WRMSR privilege (CPL 0).  Must not be called from user mode.
/// The caller is responsible for ensuring `base` is a valid canonical address
/// (bits 63:47 must all be identical — all 0 or all 1) when the value will
/// eventually be used by user-mode code; an invalid address triggers a #GP on
/// first access.
#[inline(always)]
pub unsafe fn write_user_fs_base(base: u64) {
    let lo = base as u32;
    let hi = (base >> 32) as u32;
    unsafe {
        asm!(
            "wrmsr",
            in("ecx") IA32_FS_BASE,
            in("eax") lo,
            in("edx") hi,
            options(nostack, preserves_flags, nomem),
        );
    }
}
