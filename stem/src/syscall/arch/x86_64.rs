use core::arch::asm;

/// Raw syscall entry point.
///
/// # Safety
/// This function executes a system call, which invokes the kernel. The caller must ensure
/// that the arguments satisfy the kernel's ABI for the specific syscall number `n`.
#[inline(always)]
pub unsafe fn raw_syscall6(
    n: u32,
    a0: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
) -> isize {
    abi::syscall_asm!(n, a0, a1, a2, a3, a4, a5)
}
