// On arm, the following registers are used for args 1-6:
// arg1: %r0
// arg2: %r1
// arg3: %r2
// arg4: %r3
// arg5: %r4
// arg6: %r5
//
// %r7 is used for the syscall number. In thumb mode, it is also used as the
// frame pointer, so we need to save/restore this register.
//
// %r0 is reused for the syscall return value.
//
// No other registers are clobbered.
use core::arch::asm;

use crate::arch::arm::Sysno;

/// Issues a raw system call with 0 arguments.
///
/// # Safety
///
/// Running a system call is inherently unsafe. It is the caller's
/// responsibility to ensure safety.
#[inline]
pub unsafe fn syscall0(n: Sysno) -> usize {
    let mut ret: usize;
    asm!(
        "movs {temp}, r7",
        "movs r7, {n}",
        "svc 0",
        "movs r7, {temp}",
        n = in(reg) n as usize,
        temp = out(reg) _,
        lateout("r0") ret,
        options(nostack)
    );
    ret
}

/// Issues a raw system call with 1 argument.
///
/// # Safety
///
/// Running a system call is inherently unsafe. It is the caller's
/// responsibility to ensure safety.
#[inline]
pub unsafe fn syscall1(n: Sysno, arg1: usize) -> usize {
    let mut ret: usize;
    asm!(
        "movs {temp}, r7",
        "movs r7, {n}",
        "svc 0",
        "movs r7, {temp}",
        n = in(reg) n as usize,
        temp = out(reg) _,
        inlateout("r0") arg1 => ret,
        options(nostack)
    );
    ret
}

/// Issues a raw system call with 2 arguments.
///
/// # Safety
///
/// Running a system call is inherently unsafe. It is the caller's
/// responsibility to ensure safety.
#[inline]
pub unsafe fn syscall2(n: Sysno, arg1: usize, arg2: usize) -> usize {
    let mut ret: usize;
    asm!(
        "movs {temp}, r7",
        "movs r7, {n}",
        "svc 0",
        "movs r7, {temp}",
        n = in(reg) n as usize,
        temp = out(reg) _,
        inlateout("r0") arg1 => ret,
        in("r1") arg2,
        options(nostack)
    );
    ret
}

/// Issues a raw system call with 3 arguments.
///
/// # Safety
///
/// Running a system call is inherently unsafe. It is the caller's
/// responsibility to ensure safety.
#[inline]
pub unsafe fn syscall3(
    n: Sysno,
    arg1: usize,
    arg2: usize,
    arg3: usize,
) -> usize {
    let mut ret: usize;
    asm!(
        "movs {temp}, r7",
        "movs r7, {n}",
        "svc 0",
        "movs r7, {temp}",
        n = in(reg) n as usize,
        temp = out(reg) _,
        inlateout("r0") arg1 => ret,
        in("r1") arg2,
        in("r2") arg3,
        options(nostack)
    );
    ret
}

/// Issues a raw system call with 4 arguments.
///
/// # Safety
///
/// Running a system call is inherently unsafe. It is the caller's
/// responsibility to ensure safety.
#[inline]
pub unsafe fn syscall4(
    n: Sysno,
    arg1: usize,
    arg2: usize,
    arg3: usize,
    arg4: usize,
) -> usize {
    let mut ret: usize;
    asm!(
        "movs {temp}, r7",
        "movs r7, {n}",
        "svc 0",
        "movs r7, {temp}",
        n = in(reg) n as usize,
        temp = out(reg) _,
        inlateout("r0") arg1 => ret,
        in("r1") arg2,
        in("r2") arg3,
        in("r3") arg4,
        options(nostack)
    );
    ret
}

/// Issues a raw system call with 5 arguments.
///
/// # Safety
///
/// Running a system call is inherently unsafe. It is the caller's
/// responsibility to ensure safety.
#[inline]
pub unsafe fn syscall5(
    n: Sysno,
    arg1: usize,
    arg2: usize,
    arg3: usize,
    arg4: usize,
    arg5: usize,
) -> usize {
    let mut ret: usize;
    asm!(
        "movs {temp}, r7",
        "movs r7, {n}",
        "svc 0",
        "movs r7, {temp}",
        n = in(reg) n as usize,
        temp = out(reg) _,
        inlateout("r0") arg1 => ret,
        in("r1") arg2,
        in("r2") arg3,
        in("r3") arg4,
        in("r4") arg5,
        options(nostack)
    );
    ret
}

/// Issues a raw system call with 6 arguments.
///
/// # Safety
///
/// Running a system call is inherently unsafe. It is the caller's
/// responsibility to ensure safety.
#[inline]
pub unsafe fn syscall6(
    n: Sysno,
    arg1: usize,
    arg2: usize,
    arg3: usize,
    arg4: usize,
    arg5: usize,
    arg6: usize,
) -> usize {
    let mut ret: usize;

    // NOTE: On ARMv4t, `movs` must be used instead of `mov`.
    asm!(
        "movs {temp}, r7",
        "movs r7, {n}",
        "svc 0",
        "movs r7, {temp}",
        n = in(reg) n as usize,
        temp = out(reg) _,
        inlateout("r0") arg1 => ret,
        in("r1") arg2,
        in("r2") arg3,
        in("r3") arg4,
        in("r4") arg5,
        in("r5") arg6,
        options(nostack)
    );
    ret
}
