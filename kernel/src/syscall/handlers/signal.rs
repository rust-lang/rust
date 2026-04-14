//! Signal-related syscall handlers.
//!
//! Implements:
//! - `sys_kill`        — send signal to process
//! - `sys_raise`       — send signal to self
//! - `sys_sigaction`   — install / query signal handler
//! - `sys_sigprocmask` — examine / change signal mask
//! - `sys_sigpending`  — return pending signals
//! - `sys_sigsuspend`  — atomically replace mask and sleep
//! - `sys_sigreturn`   — return from signal handler (arch-specific)
//! - `sys_alarm`       — set/cancel SIGALRM timer
//! - `sys_pause`       — sleep until a signal arrives
//! - `sys_setpgid`     — set process group ID
//! - `sys_getpgrp`     — get current process group ID
//! - `sys_setsid`      — create a new session

use super::copyin;
use crate::sched;
use crate::syscall::validate::validate_user_range;
use abi::errors::{Errno, SysResult};
use abi::signal::{SigAction, SigSet, SIGKILL, SIGSTOP, sig_how};
use core::mem::size_of;

// ── kill ─────────────────────────────────────────────────────────────────────

/// `kill(pid, sig)` — send signal `sig` to the process with PID `pid`.
///
/// - `pid > 0`  → signal exactly that PID
/// - `pid == 0` → signal the calling process's process group
/// - `pid < -1` → signal all processes in group `-pid`
/// - `pid == -1` is currently not implemented
/// - `sig == 0` → permission/existence check only (no signal sent)
pub fn sys_kill(pid_raw: usize, sig_raw: usize) -> SysResult<usize> {
    let pid = pid_raw as isize as i64;
    let sig = sig_raw as u8;

    // Validate signal number.
    if sig >= abi::signal::NSIG && sig != 0 {
        return Err(Errno::EINVAL);
    }

    if pid > 0 {
        let target_pid = pid as u32;
        // Existence check (sig == 0).
        if sig == 0 {
            let exists = sched::list_processes_current()
                .iter()
                .any(|s| s.pid == target_pid);
            if exists { Ok(0) } else { Err(Errno::ESRCH) }
        } else {
            if crate::signal::send_signal_to_process(target_pid, sig) {
                Ok(0)
            } else {
                Err(Errno::ESRCH)
            }
        }
    } else if pid == 0 {
        let pinfo = sched::process_info_current().ok_or(Errno::ESRCH)?;
        let pgid = pinfo.lock().pgid;
        let delivered = if sig == 0 {
            crate::signal::send_signal_to_group(pgid, 0)
        } else {
            crate::signal::send_signal_to_group(pgid, sig)
        };
        if delivered == 0 {
            Err(Errno::ESRCH)
        } else {
            Ok(0)
        }
    } else if pid < -1 {
        let pgid = (-pid) as u32;
        let delivered = if sig == 0 {
            crate::signal::send_signal_to_group(pgid, 0)
        } else {
            crate::signal::send_signal_to_group(pgid, sig)
        };
        if delivered == 0 {
            Err(Errno::ESRCH)
        } else {
            Ok(0)
        }
    } else {
        // pid == -1 (broadcast) is not implemented yet.
        Err(Errno::EPERM)
    }
}

/// `setpgid(pid, pgid)` — set the process group ID for `pid`.
pub fn sys_setpgid(pid_raw: usize, pgid_raw: usize) -> SysResult<usize> {
    let pid = pid_raw as isize as i64;
    let pgid = pgid_raw as isize as i64;
    crate::signal::setpgid_current(pid, pgid)?;
    Ok(0)
}

/// `getpgrp()` — return the caller's process group ID.
pub fn sys_getpgrp() -> SysResult<usize> {
    Ok(crate::signal::getpgrp_current()? as usize)
}

/// `setsid()` — create a new session and return the new session ID.
pub fn sys_setsid() -> SysResult<usize> {
    Ok(crate::signal::setsid_current()? as usize)
}

// ── raise ─────────────────────────────────────────────────────────────────────

/// `raise(sig)` — send `sig` to the calling thread.
pub fn sys_raise(sig_raw: usize) -> SysResult<usize> {
    let sig = sig_raw as u8;
    if sig == 0 || sig >= abi::signal::NSIG {
        return Err(Errno::EINVAL);
    }
    let tid = unsafe { sched::current_tid_current() };
    crate::signal::send_signal_to_thread(tid, sig);
    Ok(0)
}

// ── sigaction ─────────────────────────────────────────────────────────────────

/// `sigaction(sig, act, oldact)` — examine and/or change the action for `sig`.
pub fn sys_sigaction(
    sig_raw: usize,
    act_ptr: usize,
    oldact_ptr: usize,
) -> SysResult<usize> {
    let sig = sig_raw as u8;
    if sig == 0 || sig >= abi::signal::NSIG {
        return Err(Errno::EINVAL);
    }
    if sig == SIGKILL || sig == SIGSTOP {
        return Err(Errno::EINVAL);
    }

    let pinfo = sched::process_info_current().ok_or(Errno::ESRCH)?;
    let mut p = pinfo.lock();

    if oldact_ptr != 0 {
        validate_user_range(oldact_ptr, size_of::<SigAction>(), true)?;
        let old = p.signals.action(sig);
        let old_bytes = unsafe {
            core::slice::from_raw_parts(&old as *const SigAction as *const u8, size_of::<SigAction>())
        };
        unsafe { super::copyout(oldact_ptr, old_bytes)?; }
    }

    if act_ptr != 0 {
        validate_user_range(act_ptr, size_of::<SigAction>(), false)?;
        let mut new_action = SigAction::default();
        let dst = unsafe {
            core::slice::from_raw_parts_mut(
                &mut new_action as *mut SigAction as *mut u8,
                size_of::<SigAction>(),
            )
        };
        unsafe { copyin(dst, act_ptr)?; }
        new_action.mask = SigSet(new_action.mask.0 & !crate::signal::UNCATCHABLE.0);
        p.signals.set_action(sig, new_action);
        if new_action.handler == abi::signal::SIG_IGN {
            p.signals.pending.remove(sig);
        }
    }

    Ok(0)
}

// ── sigprocmask ───────────────────────────────────────────────────────────────

/// `sigprocmask(how, set, oldset)` — examine and/or change the calling
/// thread's signal mask.
pub fn sys_sigprocmask(
    how_raw: usize,
    set_ptr: usize,
    oldset_ptr: usize,
) -> SysResult<usize> {
    let how = how_raw as u32;
    let current_mask = sched::get_signal_mask_current();

    if oldset_ptr != 0 {
        validate_user_range(oldset_ptr, size_of::<SigSet>(), true)?;
        let mask_bytes = unsafe {
            core::slice::from_raw_parts(&current_mask as *const SigSet as *const u8, size_of::<SigSet>())
        };
        unsafe { super::copyout(oldset_ptr, mask_bytes)?; }
    }

    if set_ptr != 0 {
        validate_user_range(set_ptr, size_of::<SigSet>(), false)?;
        let mut new_set = SigSet::default();
        let dst = unsafe {
            core::slice::from_raw_parts_mut(
                &mut new_set as *mut SigSet as *mut u8,
                size_of::<SigSet>(),
            )
        };
        unsafe { copyin(dst, set_ptr)?; }
        let filtered = SigSet(new_set.0 & !crate::signal::UNCATCHABLE.0);
        let new_mask = match how {
            sig_how::SIG_BLOCK => SigSet(current_mask.0 | filtered.0),
            sig_how::SIG_UNBLOCK => SigSet(current_mask.0 & !filtered.0),
            sig_how::SIG_SETMASK => filtered,
            _ => return Err(Errno::EINVAL),
        };
        sched::set_signal_mask_current(new_mask);
    }

    Ok(0)
}

// ── sigpending ────────────────────────────────────────────────────────────────

/// `sigpending(set)` — fill `*set` with the set of blocked pending signals.
pub fn sys_sigpending(set_ptr: usize) -> SysResult<usize> {
    validate_user_range(set_ptr, size_of::<SigSet>(), true)?;

    let mask = sched::get_signal_mask_current();
    let thread_pending = sched::get_thread_pending_current();
    let proc_pending = sched::process_info_current()
        .map(|p| p.lock().signals.pending)
        .unwrap_or(SigSet::EMPTY);

    // "pending" means blocked AND pending.
    let pending = (thread_pending.union(proc_pending)).intersection(mask);

    let pending_bytes = unsafe {
        core::slice::from_raw_parts(&pending as *const SigSet as *const u8, size_of::<SigSet>())
    };
    unsafe { super::copyout(set_ptr, pending_bytes)?; }
    Ok(0)
}

// ── sigsuspend ────────────────────────────────────────────────────────────────

/// `sigsuspend(mask)` — atomically replace the thread signal mask with `*mask`
/// and suspend until a signal that is not blocked by the new mask arrives.
/// Returns `EINTR` when a signal is caught.
pub fn sys_sigsuspend(mask_ptr: usize) -> SysResult<usize> {
    validate_user_range(mask_ptr, size_of::<SigSet>(), false)?;

    let mut new_mask = SigSet::default();
    let dst = unsafe {
        core::slice::from_raw_parts_mut(
            &mut new_mask as *mut SigSet as *mut u8,
            size_of::<SigSet>(),
        )
    };
    unsafe { copyin(dst, mask_ptr)?; }
    let new_mask = SigSet(new_mask.0 & !crate::signal::UNCATCHABLE.0);

    // Save the old mask, install the new one.
    let old_mask = sched::get_signal_mask_current();
    sched::set_signal_mask_current(new_mask);

    // Block until a deliverable signal arrives.
    loop {
        let thread_pending = sched::get_thread_pending_current();
        let proc_pending = sched::process_info_current()
            .map(|p| p.lock().signals.pending)
            .unwrap_or(SigSet::EMPTY);
        let all = thread_pending.union(proc_pending);
        let deliverable = all.difference(new_mask)
            .union(all.intersection(crate::signal::UNCATCHABLE));
        if !deliverable.is_empty() {
            break;
        }
        unsafe { sched::hooks::block_current_erased() };
    }

    // Restore old mask (delivery path will update it as needed).
    sched::set_signal_mask_current(old_mask);

    Err(Errno::EINTR)
}

// ── sigreturn ─────────────────────────────────────────────────────────────────

/// `sigreturn()` — return from a signal handler.
///
/// The actual register restoration is done by the arch-specific
/// `sys_sigreturn_inner` that mutates the saved trap frame.
///
/// # Safety
///
/// The `frame_ptr` argument is the raw kernel stack pointer.
pub unsafe fn sys_sigreturn(frame_ptr: usize) -> SysResult<usize> {
    unsafe { crate::signal::delivery::sys_sigreturn_inner(frame_ptr as *mut u8) }
}

// ── alarm ─────────────────────────────────────────────────────────────────────

/// `alarm(seconds)` — schedule SIGALRM delivery after `seconds` seconds.
pub fn sys_alarm(seconds: usize) -> SysResult<usize> {
    let pinfo = sched::process_info_current().ok_or(Errno::ESRCH)?;
    let mut p = pinfo.lock();

    let now = crate::sched::TICK_COUNT.load(core::sync::atomic::Ordering::Relaxed);
    let ticks_per_sec = crate::time::SCHED_TICK_HZ;

    let remaining = if p.signals.alarm_deadline > now {
        let ticks_left = p.signals.alarm_deadline - now;
        ((ticks_left + ticks_per_sec - 1) / ticks_per_sec) as usize
    } else {
        0
    };

    if seconds == 0 {
        p.signals.alarm_deadline = 0;
    } else {
        p.signals.alarm_deadline = now + (seconds as u64) * ticks_per_sec;
    }

    Ok(remaining)
}

// ── pause ─────────────────────────────────────────────────────────────────────

/// `pause()` — suspend until a signal arrives whose disposition is not SIG_IGN.
/// Always returns `EINTR`.
pub fn sys_pause() -> SysResult<usize> {
    loop {
        let mask = sched::get_signal_mask_current();
        let thread_pending = sched::get_thread_pending_current();
        let proc_pending = sched::process_info_current()
            .map(|p| p.lock().signals.pending)
            .unwrap_or(SigSet::EMPTY);
        let all = thread_pending.union(proc_pending);
        let deliverable = all.difference(mask)
            .union(all.intersection(crate::signal::UNCATCHABLE));
        if !deliverable.is_empty() {
            break;
        }
        unsafe { sched::hooks::block_current_erased() };
    }
    Err(Errno::EINTR)
}

// No extra helpers needed — this module uses `super::copyout` and `super::copyin`
// from `crate::syscall::validate` (re-exported in `handlers/mod.rs`).

