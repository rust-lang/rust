//! POSIX-compatible signal definitions shared between kernel and userspace.
//!
//! This module provides:
//! - Standard signal number constants (`SIG_*`)
//! - Signal set (`SigSet`) — a 64-bit bitmask (supports signal numbers 1–63)
//! - `SigAction` — the per-process disposition record written by `sigaction(2)`
//! - Helper constants for `SigAction::flags` and `sigprocmask(2)` `how` values

// ── Signal numbers ────────────────────────────────────────────────────────────

pub const SIGHUP: u8 = 1;
pub const SIGINT: u8 = 2;
pub const SIGQUIT: u8 = 3;
pub const SIGILL: u8 = 4;
pub const SIGTRAP: u8 = 5;
pub const SIGABRT: u8 = 6;
pub const SIGBUS: u8 = 7;
pub const SIGFPE: u8 = 8;
pub const SIGKILL: u8 = 9;
pub const SIGUSR1: u8 = 10;
pub const SIGSEGV: u8 = 11;
pub const SIGUSR2: u8 = 12;
pub const SIGPIPE: u8 = 13;
pub const SIGALRM: u8 = 14;
pub const SIGTERM: u8 = 15;
pub const SIGSTKFLT: u8 = 16;
pub const SIGCHLD: u8 = 17;
pub const SIGCONT: u8 = 18;
pub const SIGSTOP: u8 = 19;
pub const SIGTSTP: u8 = 20;
pub const SIGTTIN: u8 = 21;
pub const SIGTTOU: u8 = 22;
pub const SIGURG: u8 = 23;
pub const SIGXCPU: u8 = 24;
pub const SIGXFSZ: u8 = 25;
pub const SIGVTALRM: u8 = 26;
pub const SIGPROF: u8 = 27;
pub const SIGWINCH: u8 = 28;
pub const SIGIO: u8 = 29;
pub const SIGPWR: u8 = 30;
pub const SIGSYS: u8 = 31;

/// One past the last standard non-realtime signal.
pub const SIGRTMIN: u8 = 32;

/// Maximum signal number (exclusive upper bound for bitmask).
pub const NSIG: u8 = 64;

// ── SIG_DFL / SIG_IGN handler values ─────────────────────────────────────────

/// Value for `SigAction::handler` meaning "use default action".
pub const SIG_DFL: usize = 0;
/// Value for `SigAction::handler` meaning "ignore this signal".
pub const SIG_IGN: usize = 1;
/// Sentinel returned by `sigaction` to indicate an error.
pub const SIG_ERR: usize = usize::MAX;

// ── SigAction::flags bits ─────────────────────────────────────────────────────

pub mod sa_flags {
    /// Restart interrupted syscalls instead of returning `EINTR`.
    pub const SA_RESTART: u32 = 1 << 0;
    /// Do not automatically block the signal while the handler runs.
    pub const SA_NODEFER: u32 = 1 << 1;
    /// Reset the disposition to `SIG_DFL` after the handler returns.
    pub const SA_RESETHAND: u32 = 1 << 2;
    /// `SIGCHLD`-specific: do not generate `SIGCHLD` for stopped children.
    pub const SA_NOCLDSTOP: u32 = 1 << 3;
    /// `SIGCHLD`-specific: do not turn stopped children into zombies.
    pub const SA_NOCLDWAIT: u32 = 1 << 4;
    /// Use an alternate signal stack (sigaltstack — not yet supported).
    pub const SA_ONSTACK: u32 = 1 << 5;
    /// Provide `siginfo_t` to handler (three-arg form — future).
    pub const SA_SIGINFO: u32 = 1 << 6;
}

// ── sigprocmask `how` constants ───────────────────────────────────────────────

pub mod sig_how {
    /// Block the signals in the provided set.
    pub const SIG_BLOCK: u32 = 0;
    /// Unblock the signals in the provided set.
    pub const SIG_UNBLOCK: u32 = 1;
    /// Replace the current mask with the provided set.
    pub const SIG_SETMASK: u32 = 2;
}

// ── SigSet ────────────────────────────────────────────────────────────────────

/// A set of signals represented as a 64-bit bitmask.
///
/// Signal *n* is present when bit *(n − 1)* is set.  Valid signal numbers are
/// in the range `[1, 63]`.  Signal 0 is never valid and the corresponding bit
/// is ignored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct SigSet(pub u64);

impl SigSet {
    /// Empty set (no signals).
    pub const EMPTY: Self = Self(0);

    /// Full set (all signals 1–63).
    pub const FULL: Self = Self(u64::MAX & !1);

    /// Returns `true` if signal `sig` is a member of this set.
    #[inline]
    pub fn contains(self, sig: u8) -> bool {
        if sig == 0 || sig >= NSIG {
            return false;
        }
        self.0 & (1u64 << (sig - 1)) != 0
    }

    /// Add signal `sig` to this set.
    #[inline]
    pub fn add(&mut self, sig: u8) {
        if sig > 0 && sig < NSIG {
            self.0 |= 1u64 << (sig - 1);
        }
    }

    /// Remove signal `sig` from this set.
    #[inline]
    pub fn remove(&mut self, sig: u8) {
        if sig > 0 && sig < NSIG {
            self.0 &= !(1u64 << (sig - 1));
        }
    }

    /// Compute the union with another set.
    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Compute the intersection with another set.
    #[inline]
    pub fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Compute the difference (signals in `self` but not in `other`).
    #[inline]
    pub fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Return the lowest-numbered pending signal, or `0` if none.
    #[inline]
    pub fn lowest(self) -> u8 {
        if self.0 == 0 {
            return 0;
        }
        // trailing_zeros gives the bit index (0-based); signal number is +1
        (self.0.trailing_zeros() as u8) + 1
    }

    /// Returns `true` if this set contains no signals.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

// ── SigAction ─────────────────────────────────────────────────────────────────

/// Per-signal action descriptor, mirrors POSIX `struct sigaction`.
///
/// `handler` is one of:
/// - [`SIG_DFL`] (0) — execute the default action
/// - [`SIG_IGN`] (1) — ignore the signal
/// - Any other value — a userspace function pointer
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct SigAction {
    /// Signal handler (function pointer, `SIG_DFL`, or `SIG_IGN`).
    pub handler: usize,
    /// Signals to mask during execution of the handler.
    pub mask: SigSet,
    /// Flags from [`sa_flags`].
    pub flags: u32,
    /// Optional trampoline pointer (restorer) — set by libc, may be zero.
    pub restorer: usize,
}

// ── WaitStatus helpers ────────────────────────────────────────────────────────

/// Encode a normal (exit-code) wait status.
#[inline]
pub fn w_exit_status(code: u8) -> i32 {
    (code as i32) << 8
}

/// Encode a signal-terminated wait status.
#[inline]
pub fn w_term_sig(sig: u8) -> i32 {
    sig as i32
}

/// Encode a stopped wait status (WUNTRACED).
#[inline]
pub fn w_stop_sig(sig: u8) -> i32 {
    0x7f | ((sig as i32) << 8)
}

/// Encode a continued wait status (WCONTINUED).
#[inline]
pub fn w_continued() -> i32 {
    0xffff
}

/// Decode the exit code from a normal termination status.
#[inline]
pub fn wexitstatus(status: i32) -> u8 {
    ((status >> 8) & 0xff) as u8
}

/// Return `true` if `status` encodes a normal exit.
#[inline]
pub fn wifexited(status: i32) -> bool {
    (status & 0x7f) == 0
}

/// Return `true` if `status` encodes a signal-terminated exit.
#[inline]
pub fn wifsignaled(status: i32) -> bool {
    ((status & 0x7f) != 0) && ((status & 0x7f) != 0x7f)
}

/// Decode the termination signal from a signalled exit status.
#[inline]
pub fn wtermsig(status: i32) -> u8 {
    (status & 0x7f) as u8
}

/// Return `true` if `status` encodes a stopped process.
#[inline]
pub fn wifstopped(status: i32) -> bool {
    (status & 0xff) == 0x7f
}

/// Decode the stopping signal from a stopped status.
#[inline]
pub fn wstopsig(status: i32) -> u8 {
    ((status >> 8) & 0xff) as u8
}

/// Return `true` if `status` encodes a continued process.
#[inline]
pub fn wifcontinued(status: i32) -> bool {
    status == 0xffff
}
