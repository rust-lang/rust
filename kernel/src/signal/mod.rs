//! Kernel signal subsystem.
//!
//! This module owns the per-process signal state: dispositions, pending sets,
//! and delivery.  Per-thread signal masks and thread-local pending signals live
//! in the `Thread` struct but are accessed through helpers here.
//!
//! # Unix legacy compatibility boundary
//!
//! The entire signal subsystem — including `ProcessSignals`, the `pgid`/`sid`
//! manipulation in `setpgid_current`/`setsid_current`, and the job-control
//! stop signals (SIGTTOU, SIGTTIN, SIGTSTP) — is **quarantined Unix legacy
//! compatibility state**.  It lives in `Process.unix_compat` and is accessed
//! only through the explicit `unix_compat.*` prefix.
//!
//! **New code must not add Unix session or process-group assumptions here.**
//! Specifically:
//!
//! - `pgid`/`sid`/`session_leader` operations belong to the `Group` domain
//!   (Phase 5) and must not be deepened.
//! - SIGTTOU/SIGTTIN/SIGTSTP job-control stop signals belong to
//!   `Group / Presence` and must not be entangled further with the core
//!   signal table.
//! - Any unavoidable Unix signal compatibility code must remain inside
//!   `Process.unix_compat.signals`; it must not leak into new top-level
//!   `Process` fields.
//!
//! See `docs/concepts/unix-compat.md` and
//! `docs/migration/process_execution_context_inventory.md` for the canonical
//! quarantine inventory and future migration plan.
//!
//! # Design notes
//!
//! - Signal numbers 1–31 are standard non-realtime signals (POSIX).
//! - SIGKILL (9) and SIGSTOP (19) cannot be caught, blocked, or ignored.
//! - Disposition, pending sets, and masks use a `u64` bitmask where bit *k*
//!   represents signal *k+1* (so signal 1 is bit 0, signal 31 is bit 30).
//! - Delivery is checked in `kernel_post_syscall_signal_check`, called from the
//!   architecture-specific syscall return path after every syscall.
//! - Signal handler invocation is architecture-specific; see `arch/delivery.rs`.

pub mod delivery;
pub mod routing;

use abi::errors::Errno;
use abi::signal::{SIG_DFL, SIG_IGN, SIGCHLD, SIGCONT, SIGKILL, SIGSTOP, SigAction, SigSet};

/// Uncatchable signals — cannot be caught, blocked, or ignored.
pub const UNCATCHABLE: SigSet = SigSet((1u64 << (SIGKILL - 1)) | (1u64 << (SIGSTOP - 1)));

/// Signals that are ignored by default (no default action kills the process).
/// SIGCHLD, SIGCONT, SIGURG, SIGWINCH are in this set.
const DEFAULT_IGNORED: SigSet = SigSet(
    (1u64 << (abi::signal::SIGCHLD - 1))
        | (1u64 << (abi::signal::SIGCONT - 1))
        | (1u64 << (abi::signal::SIGURG - 1))
        | (1u64 << (abi::signal::SIGWINCH - 1)),
);

/// Default action classifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefaultAction {
    /// Terminate the process.
    Terminate,
    /// Terminate the process and generate a core dump (stub — behaves like Terminate).
    CoreDump,
    /// Ignore the signal.
    Ignore,
    /// Stop the process (job control).
    Stop,
    /// Continue a stopped process.
    Continue,
}

/// Return the default action for signal `sig`.
pub fn default_action(sig: u8) -> DefaultAction {
    use abi::signal::*;
    match sig {
        SIGHUP | SIGINT | SIGKILL | SIGPIPE | SIGALRM | SIGTERM | SIGUSR1 | SIGUSR2 | SIGSTKFLT
        | SIGPWR => DefaultAction::Terminate,

        SIGQUIT | SIGILL | SIGTRAP | SIGABRT | SIGBUS | SIGFPE | SIGSEGV | SIGXCPU | SIGXFSZ
        | SIGSYS => DefaultAction::CoreDump,

        SIGCHLD | SIGCONT | SIGURG | SIGWINCH => DefaultAction::Ignore,

        SIGSTOP | SIGTSTP | SIGTTIN | SIGTTOU => DefaultAction::Stop,

        _ => DefaultAction::Terminate,
    }
}

// ── Per-process signal state ──────────────────────────────────────────────────

/// The signal state owned by a single process (thread group).
///
/// Protected by the `Process` mutex.
pub struct ProcessSignals {
    /// Per-signal action table; index = signal number - 1.
    pub actions: [SigAction; 64],
    /// Process-directed pending signals (not yet assigned to a thread).
    pub pending: SigSet,
    /// True while the process is stopped (SIGSTOP received, SIGCONT not yet).
    pub stopped: bool,
    /// If > 0, an `alarm(2)` is armed; expires at this monotonic tick.
    pub alarm_deadline: u64,
    /// Waitpid notification: set of (child_pid, exit_status) pairs waiting
    /// for `waitpid` to consume them.  We reuse `Process.children_done` below;
    /// this field is kept here for conceptual clarity.
    pub _reserved: u64,
}

impl ProcessSignals {
    pub fn new() -> Self {
        Self {
            actions: [SigAction::default(); 64],
            pending: SigSet::EMPTY,
            stopped: false,
            alarm_deadline: 0,
            _reserved: 0,
        }
    }

    /// Return the `SigAction` for signal `sig`.
    pub fn action(&self, sig: u8) -> SigAction {
        if sig == 0 || sig >= abi::signal::NSIG {
            return SigAction::default();
        }
        self.actions[(sig - 1) as usize]
    }

    /// Update the disposition for `sig`, returning the old action.
    ///
    /// Silently ignores attempts to change SIGKILL / SIGSTOP.
    pub fn set_action(&mut self, sig: u8, new: SigAction) -> SigAction {
        if sig == 0 || sig >= abi::signal::NSIG {
            return SigAction::default();
        }
        if UNCATCHABLE.contains(sig) {
            return SigAction::default();
        }
        let old = self.actions[(sig - 1) as usize];
        self.actions[(sig - 1) as usize] = new;
        old
    }

    /// Queue a signal to the process-level pending set.
    pub fn post(&mut self, sig: u8) {
        if sig == 0 || sig >= abi::signal::NSIG {
            return;
        }
        // Ignored signals (disposition = SIG_IGN) are silently dropped.
        let a = self.action(sig);
        if a.handler == SIG_IGN && !UNCATCHABLE.contains(sig) {
            return;
        }
        self.pending.add(sig);
    }

    /// Dequeue a deliverable signal given the thread's blocked mask.
    ///
    /// Returns the signal number (1–63) and the corresponding action, or `None`
    /// if no deliverable signal exists.  SIGKILL/SIGSTOP bypass the mask.
    pub fn take_deliverable(&mut self, mask: SigSet) -> Option<(u8, SigAction)> {
        // Try process-level pending first.
        let mut candidate = self.pending.difference(mask);
        // Force-deliver uncatchable signals regardless of mask.
        candidate = candidate.union(self.pending.intersection(UNCATCHABLE));
        let sig = candidate.lowest();
        if sig == 0 {
            return None;
        }
        self.pending.remove(sig);
        let action = self.action(sig);
        Some((sig, action))
    }

    /// Clear all pending signals — called on exec.
    pub fn clear_pending(&mut self) {
        self.pending = SigSet::EMPTY;
    }

    /// Reset all caught dispositions to SIG_DFL — called on exec.
    pub fn reset_for_exec(&mut self) {
        for a in &mut self.actions {
            if a.handler != SIG_IGN {
                *a = SigAction::default(); // handler = SIG_DFL
            }
        }
        self.pending = SigSet::EMPTY;
    }
}

impl Default for ProcessSignals {
    fn default() -> Self {
        Self::new()
    }
}

// ── Per-thread signal state ───────────────────────────────────────────────────

/// The signal state owned by a single thread.
#[derive(Clone, Copy, Debug, Default)]
pub struct ThreadSignals {
    /// The set of signals blocked (masked) by this thread.
    ///
    /// SIGKILL and SIGSTOP are always removed from this mask before use.
    pub mask: SigSet,
    /// Thread-directed pending signals.
    pub pending: SigSet,
}

impl ThreadSignals {
    pub fn new() -> Self {
        Self {
            mask: SigSet::EMPTY,
            pending: SigSet::EMPTY,
        }
    }

    /// Apply the effective mask (strip SIGKILL/SIGSTOP which cannot be blocked).
    #[inline]
    pub fn effective_mask(&self) -> SigSet {
        SigSet(self.mask.0 & !UNCATCHABLE.0)
    }
}

// ── Generation helpers ────────────────────────────────────────────────────────

/// Post signal `sig` to the process identified by `pid`.
///
/// Returns `true` if the process was found and the signal delivered to its
/// queue.  The signal may still be masked or ignored.
///
/// # Routing
///
/// This function now routes through the typed signal routing layer
/// ([`routing::route_signal`]) so that targeting is explicit and observable.
/// Compatibility semantics (pending set, masks, disposition, handler
/// invocation) are fully preserved through the canonical
/// [`routing::deliver_to_recipient`] hook.
///
/// # Permissions (simplified)
///
/// Any process can send SIGCONT to a process in the same session.  For now
/// ThingOS grants kill permission freely (no UID/GID checks yet).
pub fn send_signal_to_process(pid: u32, sig: u8) -> bool {
    use routing::{SignalRoute, SignalTargetKind, SignalDeliveryOutcome};
    let route = SignalRoute {
        signal: sig,
        sender_tid: Some(unsafe { crate::sched::current_tid_current() }),
        target_kind: SignalTargetKind::Process,
        target_id: pid as u64,
    };
    let report = routing::route_signal(route);
    report.succeeded > 0
        || report.failures.iter().all(|(_, o)| *o != SignalDeliveryOutcome::RecipientNotFound)
}

/// Post signal `sig` to the process containing thread `tid`.
///
/// Routes through the typed signal routing layer for explicit, observable
/// targeting.  Compatibility semantics are preserved through the canonical
/// [`routing::deliver_to_thread_recipient`] hook.
pub fn send_signal_to_thread(tid: u64, sig: u8) {
    use routing::{SignalRoute, SignalTargetKind};
    let route = SignalRoute {
        signal: sig,
        sender_tid: Some(unsafe { crate::sched::current_tid_current() }),
        target_kind: SignalTargetKind::Thread,
        target_id: tid,
    };
    routing::route_signal(route);
}

/// Deliver SIGCHLD to the parent of the process with PID `child_pid`.
///
/// Called from the scheduler exit path when a child process exits or stops.
pub fn notify_parent_sigchld(ppid: u32, _child_pid: u32, _status: i32) {
    if ppid == 0 {
        return;
    }
    send_signal_to_process(ppid, SIGCHLD);
}

fn process_info_for_pid(pid: u32) -> Option<alloc::sync::Arc<spin::Mutex<crate::task::Process>>> {
    crate::sched::process_info_for_tid_current(pid as u64)
}

fn list_unique_processes() -> alloc::vec::Vec<alloc::sync::Arc<spin::Mutex<crate::task::Process>>> {
    let mut seen = alloc::collections::BTreeSet::new();
    let mut out = alloc::vec::Vec::new();
    for snap in crate::sched::list_processes_current() {
        if seen.insert(snap.pid)
            && let Some(pinfo) = process_info_for_pid(snap.pid)
        {
            out.push(pinfo);
        }
    }
    out
}

pub fn process_group_exists_in_session(pgid: u32, sid: u32) -> bool {
    for pinfo in list_unique_processes() {
        let p = pinfo.lock();
        if p.unix_compat.pgid == pgid && p.unix_compat.sid == sid {
            return true;
        }
    }
    false
}

pub fn send_signal_to_group(pgid: u32, sig: u8) -> usize {
    use routing::{SignalRoute, SignalTargetKind};
    let route = SignalRoute {
        signal: sig,
        sender_tid: Some(unsafe { crate::sched::current_tid_current() }),
        target_kind: SignalTargetKind::ProcessGroup,
        target_id: pgid as u64,
    };
    let report = routing::route_signal(route);
    // For the sig==0 existence check, targeted counts as delivered.
    if sig == 0 { report.targeted } else { report.succeeded }
}

pub fn getpgrp_current() -> Result<u32, Errno> {
    let pinfo = crate::sched::process_info_current().ok_or(Errno::ESRCH)?;
    Ok(pinfo.lock().unix_compat.pgid)
}

pub fn setpgid_current(pid: i64, pgid: i64) -> Result<(), Errno> {
    let caller_info = crate::sched::process_info_current().ok_or(Errno::ESRCH)?;
    let (caller_pid, caller_sid) = {
        let caller = caller_info.lock();
        (caller.pid, caller.unix_compat.sid)
    };

    if pid < 0 || pgid < 0 {
        return Err(Errno::EINVAL);
    }

    let target_pid = if pid == 0 { caller_pid } else { pid as u32 };
    let new_pgid = if pgid == 0 { target_pid } else { pgid as u32 };
    if new_pgid == 0 {
        return Err(Errno::EINVAL);
    }

    let target_info = process_info_for_pid(target_pid).ok_or(Errno::ESRCH)?;
    {
        let target = target_info.lock();
        if target.pid != caller_pid && target.lifecycle.ppid != caller_pid {
            return Err(Errno::EPERM);
        }
        if target.unix_compat.sid != caller_sid {
            return Err(Errno::EPERM);
        }
        if target.unix_compat.session_leader {
            return Err(Errno::EPERM);
        }
    }

    if new_pgid != target_pid && !process_group_exists_in_session(new_pgid, caller_sid) {
        return Err(Errno::EPERM);
    }

    target_info.lock().unix_compat.pgid = new_pgid;
    Ok(())
}

pub fn setsid_current() -> Result<u32, Errno> {
    let pinfo = crate::sched::process_info_current().ok_or(Errno::ESRCH)?;
    {
        let p = pinfo.lock();
        if p.unix_compat.pgid == p.pid {
            return Err(Errno::EPERM);
        }
    }

    let mut p = pinfo.lock();
    p.unix_compat.sid = p.pid;
    p.unix_compat.pgid = p.pid;
    p.unix_compat.session_leader = true;
    Ok(p.unix_compat.sid)
}
