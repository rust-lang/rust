//! Raw syscall functions.
#![unstable(feature = "linux_syscall", issue = "63748")]
#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]

#[cfg(target_arch = "x86")]
#[path="x86.rs"] mod platform;

#[cfg(target_arch = "x86_64")]
#[path="x86_64.rs"] mod platform;

#[cfg(target_arch = "aarch64")]
#[path="aarch64"] mod platform;

/// Execute syscall with 0 arguments.
#[unstable(feature = "linux_syscall", issue = "63748")]
#[inline(always)]
pub unsafe fn syscall0(n: usize) -> usize {
    platform::syscall0(n)
}

/// Execute syscall with 1 argument.
#[unstable(feature = "linux_syscall", issue = "63748")]
#[inline(always)]
pub unsafe fn syscall1(n: usize, a1: usize) -> usize {
    platform::syscall1(n, a1)
}

/// Execute syscall with 2 arguments.
#[unstable(feature = "linux_syscall", issue = "63748")]
#[inline(always)]
pub unsafe fn syscall2(n: usize, a1: usize, a2: usize) -> usize {
    platform::syscall2(n, a1, a2)
}

/// Execute syscall with 3 arguments.
#[unstable(feature = "linux_syscall", issue = "63748")]
#[inline(always)]
pub unsafe fn syscall3(n: usize, a1: usize, a2: usize, a3: usize) -> usize {
    platform::syscall3(n, a1, a2, a3)
}

/// Execute syscall with 4 arguments.
#[unstable(feature = "linux_syscall", issue = "63748")]
#[inline(always)]
pub unsafe fn syscall4(
    n: usize, a1: usize, a2: usize, a3: usize, a4: usize,
) -> usize {
    platform::syscall4(n, a1, a2, a3, a4)
}

/// Execute syscall with 5 arguments.
#[unstable(feature = "linux_syscall", issue = "63748")]
#[inline(always)]
pub unsafe fn syscall5(
    n: usize, a1: usize, a2: usize, a3: usize, a4: usize, a5: usize,
) -> usize {
    platform::syscall5(n, a1, a2, a3, a4, a5)
}

/// Execute syscall with 6 arguments.
#[unstable(feature = "linux_syscall", issue = "63748")]
#[inline(always)]
pub unsafe fn syscall6(
    n: usize, a1: usize, a2: usize, a3: usize, a4: usize, a5: usize, a6: usize,
) -> usize {
    platform::syscall5(n, a1, a2, a3, a4, a5, a6)
}
