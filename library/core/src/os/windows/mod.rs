//! Platform-specific extensions to `core` for Windows platforms.

#![unstable(issue = "none", feature = "std_internals")]

use crate::cfg_select;

const FAST_FAIL_FATAL_APP_EXIT: u32 = 7u32;

/// Use `__fastfail` to abort the process
///
/// In Windows 8 and later, this will terminate the process immediately without
/// running any in-process exception handlers. In earlier versions of Windows,
/// this sequence of instructions will be treated as an access violation, which
/// will still terminate the process but might run some exception handlers.
///
/// <https://docs.microsoft.com/en-us/cpp/intrinsics/fastfail>
#[cfg(all(
    not(miri),
    any(
        any(target_arch = "x86", target_arch = "x86_64"),
        all(target_arch = "arm", target_feature = "thumb-mode"),
        any(target_arch = "aarch64", target_arch = "arm64ec")
    )
))]
pub fn fastfail() -> ! {
    // SAFETY: These assembly instructions are always safe to call and will result in the documented behavior.
    unsafe {
        cfg_select! {
            any(target_arch = "x86", target_arch = "x86_64") => {
                core::arch::asm!("int 0x29", in("ecx") FAST_FAIL_FATAL_APP_EXIT, options(noreturn, nostack));
            }
            all(target_arch = "arm", target_feature = "thumb-mode") => {
                core::arch::asm!(".inst 0xDEFB", in("r0") FAST_FAIL_FATAL_APP_EXIT, options(noreturn, nostack));
            }
            any(target_arch = "aarch64", target_arch = "arm64ec") => {
                core::arch::asm!("brk 0xF003", in("x0") FAST_FAIL_FATAL_APP_EXIT, options(noreturn, nostack));
            }
            _ => {}
        }
    }
}
