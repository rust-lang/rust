//! Userspace wrappers for POSIX-compatible signal syscalls.

use abi::errors::{Errno, SysResult, errno};
use abi::syscall::{
    SYS_ALARM, SYS_GETPGRP, SYS_KILL, SYS_PAUSE, SYS_RAISE, SYS_SETPGID, SYS_SETSID,
    SYS_SIGACTION, SYS_SIGPENDING, SYS_SIGPROCMASK, SYS_SIGSUSPEND,
};
use abi::signal::{SigAction, SigSet, sig_how};

use super::arch::raw_syscall6;

/// Send signal `sig` to the process identified by `pid`.
///
/// `pid > 0`  — signal that process.
/// `pid == 0` — signal the caller's own process.
/// `pid < 0`  — signal a process group (stub: returns `EPERM`).
/// `sig == 0` — existence/permission check only.
pub fn kill(pid: i32, sig: u8) -> SysResult<()> {
    let ret = unsafe { raw_syscall6(SYS_KILL, pid as usize, sig as usize, 0, 0, 0, 0) };
    errno(ret).map(|_| ())
}

/// Send signal `sig` to the calling thread.
pub fn raise(sig: u8) -> SysResult<()> {
    let ret = unsafe { raw_syscall6(SYS_RAISE, sig as usize, 0, 0, 0, 0, 0) };
    errno(ret).map(|_| ())
}

/// Examine and/or change the action for signal `sig`.
pub fn sigaction(
    sig: u8,
    act: Option<&SigAction>,
    oldact: Option<&mut SigAction>,
) -> SysResult<()> {
    let act_ptr = act.map(|a| a as *const SigAction as usize).unwrap_or(0);
    let oldact_ptr = oldact.map(|a| a as *mut SigAction as usize).unwrap_or(0);
    let ret = unsafe { raw_syscall6(SYS_SIGACTION, sig as usize, act_ptr, oldact_ptr, 0, 0, 0) };
    errno(ret).map(|_| ())
}

/// Examine and/or change the calling thread's signal mask.
pub fn sigprocmask(
    how: u32,
    set: Option<&SigSet>,
    oldset: Option<&mut SigSet>,
) -> SysResult<()> {
    let set_ptr = set.map(|s| s as *const SigSet as usize).unwrap_or(0);
    let oldset_ptr = oldset.map(|s| s as *mut SigSet as usize).unwrap_or(0);
    let ret =
        unsafe { raw_syscall6(SYS_SIGPROCMASK, how as usize, set_ptr, oldset_ptr, 0, 0, 0) };
    errno(ret).map(|_| ())
}

/// Block the given set of signals.
#[inline]
pub fn sig_block(set: &SigSet) -> SysResult<()> {
    sigprocmask(sig_how::SIG_BLOCK, Some(set), None)
}

/// Unblock the given set of signals.
#[inline]
pub fn sig_unblock(set: &SigSet) -> SysResult<()> {
    sigprocmask(sig_how::SIG_UNBLOCK, Some(set), None)
}

/// Replace the signal mask entirely.
#[inline]
pub fn sig_setmask(set: &SigSet) -> SysResult<()> {
    sigprocmask(sig_how::SIG_SETMASK, Some(set), None)
}

/// Return the set of signals that are blocked and pending for the calling thread.
pub fn sigpending() -> SysResult<SigSet> {
    let mut set = SigSet::EMPTY;
    let ret = unsafe {
        raw_syscall6(SYS_SIGPENDING, &mut set as *mut SigSet as usize, 0, 0, 0, 0, 0)
    };
    errno(ret).map(|_| set)
}

/// Atomically replace the signal mask with `mask` and wait for a signal.
///
/// Always returns `Err(Errno::EINTR)`.
pub fn sigsuspend(mask: &SigSet) -> Errno {
    let ret = unsafe {
        raw_syscall6(SYS_SIGSUSPEND, mask as *const SigSet as usize, 0, 0, 0, 0, 0)
    };
    errno(ret).map(|_| ()).unwrap_err()
}

/// Schedule `SIGALRM` delivery after `seconds` seconds.
///
/// Returns the number of seconds until the previously-scheduled alarm fires.
pub fn alarm(seconds: u32) -> u32 {
    let ret = unsafe { raw_syscall6(SYS_ALARM, seconds as usize, 0, 0, 0, 0, 0) };
    ret.max(0) as u32
}

/// Suspend until a signal arrives (any non-ignored signal).
///
/// Always returns `EINTR`.
pub fn pause() -> Errno {
    let ret = unsafe { raw_syscall6(SYS_PAUSE, 0, 0, 0, 0, 0, 0) };
    errno(ret).map(|_| ()).unwrap_err()
}

/// Set process group ID for `pid` (`0` means caller).
pub fn setpgid(pid: i32, pgid: i32) -> SysResult<()> {
    let ret = unsafe { raw_syscall6(SYS_SETPGID, pid as usize, pgid as usize, 0, 0, 0, 0) };
    errno(ret).map(|_| ())
}

/// Return the caller's process group ID.
pub fn getpgrp() -> SysResult<i32> {
    let ret = unsafe { raw_syscall6(SYS_GETPGRP, 0, 0, 0, 0, 0, 0) };
    errno(ret).map(|v| v as i32)
}

/// Create a new session and return the new session ID.
pub fn setsid() -> SysResult<i32> {
    let ret = unsafe { raw_syscall6(SYS_SETSID, 0, 0, 0, 0, 0, 0) };
    errno(ret).map(|v| v as i32)
}
