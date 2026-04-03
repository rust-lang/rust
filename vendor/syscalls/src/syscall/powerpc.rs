// PowerPC uses the following registers for args 1-6:
//
// arg1: r3
// arg2: r4
// arg3: r5
// arg4: r6
// arg5: r7
// arg6: r8
//
// Register r0 specifies the syscall number.
// Register r3 is also used for the return value.
// Registers r0, r3-r12, and cr0 are always clobbered.
//
// The `sc` instruction is used to perform the syscall. If successful, then it
// sets the summary overflow bit (S0) in field 0 of the condition register
// (cr0). This is then used to decide if the return value should be negated.
use core::arch::asm;

use crate::arch::powerpc::Sysno;

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
        "sc",
        "bns 1f",
        "neg 3, 3",
        "1:",
        inlateout("r0") n as usize => _,
        lateout("r3") ret,
        lateout("r4") _,
        lateout("r5") _,
        lateout("r6") _,
        lateout("r7") _,
        lateout("r8") _,
        lateout("r9") _,
        lateout("r10") _,
        lateout("r11") _,
        lateout("r12") _,
        lateout("cr0") _,
        options(nostack, preserves_flags)
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
        "sc",
        "bns 1f",
        "neg 3, 3",
        "1:",
        inlateout("r0") n as usize => _,
        inlateout("r3") arg1 => ret,
        lateout("r4") _,
        lateout("r5") _,
        lateout("r6") _,
        lateout("r7") _,
        lateout("r8") _,
        lateout("r9") _,
        lateout("r10") _,
        lateout("r11") _,
        lateout("r12") _,
        lateout("cr0") _,
        options(nostack, preserves_flags)
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
        "sc",
        "bns 1f",
        "neg 3, 3",
        "1:",
        inlateout("r0") n as usize => _,
        inlateout("r3") arg1 => ret,
        inlateout("r4") arg2 => _,
        lateout("r5") _,
        lateout("r6") _,
        lateout("r7") _,
        lateout("r8") _,
        lateout("r9") _,
        lateout("r10") _,
        lateout("r11") _,
        lateout("r12") _,
        lateout("cr0") _,
        options(nostack, preserves_flags)
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
        "sc",
        "bns 1f",
        "neg 3, 3",
        "1:",
        inlateout("r0") n as usize => _,
        inlateout("r3") arg1 => ret,
        inlateout("r4") arg2 => _,
        inlateout("r5") arg3 => _,
        lateout("r6") _,
        lateout("r7") _,
        lateout("r8") _,
        lateout("r9") _,
        lateout("r10") _,
        lateout("r11") _,
        lateout("r12") _,
        lateout("cr0") _,
        options(nostack, preserves_flags)
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
        "sc",
        "bns 1f",
        "neg 3, 3",
        "1:",
        inlateout("r0") n as usize => _,
        inlateout("r3") arg1 => ret,
        inlateout("r4") arg2 => _,
        inlateout("r5") arg3 => _,
        inlateout("r6") arg4 => _,
        lateout("r7") _,
        lateout("r8") _,
        lateout("r9") _,
        lateout("r10") _,
        lateout("r11") _,
        lateout("r12") _,
        lateout("cr0") _,
        options(nostack, preserves_flags)
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
        "sc",
        "bns 1f",
        "neg 3, 3",
        "1:",
        inlateout("r0") n as usize => _,
        inlateout("r3") arg1 => ret,
        inlateout("r4") arg2 => _,
        inlateout("r5") arg3 => _,
        inlateout("r6") arg4 => _,
        inlateout("r7") arg5 => _,
        lateout("r8") _,
        lateout("r9") _,
        lateout("r10") _,
        lateout("r11") _,
        lateout("r12") _,
        lateout("cr0") _,
        options(nostack, preserves_flags)
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
    asm!(
        "sc",
        "bns 1f",
        "neg 3, 3",
        "1:",
        inlateout("r0") n as usize => _,
        inlateout("r3") arg1 => ret,
        inlateout("r4") arg2 => _,
        inlateout("r5") arg3 => _,
        inlateout("r6") arg4 => _,
        inlateout("r7") arg5 => _,
        inlateout("r8") arg6 => _,
        lateout("r9") _,
        lateout("r10") _,
        lateout("r11") _,
        lateout("r12") _,
        lateout("cr0") _,
        options(nostack, preserves_flags)
    );
    ret
}
