//! ThingOS PAL — core platform primitives.
//!
//! This is a stub implementation. Functions will be fleshed out as the
//! Thing-OS userspace runtime (stem) exposes the necessary syscalls.

use crate::io as std_io;

// SAFETY: must be called only once during runtime initialization.
// NOTE: this is not guaranteed to run, for example when Rust code is called externally.
pub unsafe fn init(_argc: isize, _argv: *const *const u8, _sigpipe: u8) {}

// SAFETY: must be called only once during runtime cleanup.
// NOTE: this is not guaranteed to run, for example when the program aborts.
pub unsafe fn cleanup() {}

#[allow(dead_code)]
pub fn unsupported<T>() -> std_io::Result<T> {
    Err(unsupported_err())
}

#[allow(dead_code)]
pub fn unsupported_err() -> std_io::Error {
    std_io::Error::UNSUPPORTED_PLATFORM
}

pub fn abort_internal() -> ! {
    core::intrinsics::abort();
}

/// Low-level syscall entry point shared across ThingOS `sys` modules.
///
/// Exposed as `pub` so that platform-specific modules such as `sys::time`,
/// `sys::args`, `sys::env`, and `sys::stdio` can call it via
/// `crate::sys::pal::raw_syscall6` without duplicating the inline-asm.
pub unsafe fn raw_syscall6(
    n: u32,
    a0: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
) -> isize {
    #[cfg(target_arch = "x86_64")]
    {
        let ret: isize;
        unsafe {
            core::arch::asm!(
                "syscall",
                inlateout("rax") n as usize => ret,
                in("rdi") a0,
                in("rsi") a1,
                in("rdx") a2,
                in("r10") a3,
                in("r8") a4,
                in("r9") a5,
                out("rcx") _,
                out("r11") _,
                options(nostack, preserves_flags)
            );
        }
        return ret;
    }

    #[cfg(target_arch = "aarch64")]
    {
        let ret: isize;
        unsafe {
            core::arch::asm!(
                "svc #0",
                inlateout("x0") a0 as isize => ret,
                in("x1") a1,
                in("x2") a2,
                in("x3") a3,
                in("x4") a4,
                in("x5") a5,
                in("x8") n,
                options(nostack, preserves_flags)
            );
        }
        return ret;
    }

    #[cfg(target_arch = "riscv64")]
    {
        let ret: isize;
        unsafe {
            core::arch::asm!(
                "ecall",
                inlateout("a0") a0 as isize => ret,
                in("a1") a1,
                in("a2") a2,
                in("a3") a3,
                in("a4") a4,
                in("a5") a5,
                in("a7") n,
                options(nostack, preserves_flags)
            );
        }
        return ret;
    }

    #[cfg(target_arch = "loongarch64")]
    {
        let ret: isize;
        unsafe {
            core::arch::asm!(
                "syscall 0",
                inlateout("$a0") a0 as isize => ret,
                in("$a1") a1,
                in("$a2") a2,
                in("$a3") a3,
                in("$a4") a4,
                in("$a5") a5,
                in("$a7") n,
                options(nostack, preserves_flags)
            );
        }
        return ret;
    }

    #[allow(unreachable_code)]
    {
        let _ = (n, a0, a1, a2, a3, a4, a5);
        -95 // ENOTSUP fallback
    }
}
