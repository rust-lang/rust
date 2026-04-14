use core::arch::asm;

/// Raw syscall entry point.
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
    let ret: isize;
    asm!(
        "svc #0",
        inlateout("x0") a0 as isize => ret,
        in("x1") a1,
        in("x2") a2,
        in("x3") a3,
        in("x4") a4,
        in("x5") a5,
        in("x8") n, // Syscall number in x8
        options(nostack, preserves_flags)
    );
    ret
}
