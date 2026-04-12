//! ThingOS time implementation.
//!
//! `Instant`    — backed by `SYS_TIME_MONOTONIC` (0x1202), returns nanoseconds as usize.
//! `SystemTime` — backed by `SYS_TIME_NOW`       (0x1203) with ClockId::Realtime (2).
//!
//! The `TimeSpec` written by `SYS_TIME_NOW` has the layout:
//!   secs:     u64 LE
//!   nanos:    u32 LE
//!   reserved: u32 LE
//! (matches abi/src/time.rs `TimeSpec`)

use crate::time::Duration;

// Syscall numbers (abi/src/numbers.rs)
const SYS_TIME_MONOTONIC: u32 = 0x1202;
const SYS_TIME_NOW: u32 = 0x1203;
const CLOCK_REALTIME: usize = 2;

/// Perform a raw syscall. Mirrors `sys/pal/thingos/common.rs`.
#[inline(always)]
unsafe fn raw_syscall6(
    n: u32,
    a0: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
) -> isize {
    // SAFETY: caller guarantees arguments are valid for this syscall.
    unsafe { crate::sys::pal::raw_syscall6(n, a0, a1, a2, a3, a4, a5) }
}

/// Query SYS_TIME_MONOTONIC — returns nanoseconds since boot as a usize.
///
/// Returns 0 on syscall failure; callers treat 0 as the epoch of the monotonic
/// clock rather than aborting, matching the behaviour of other embedded PALs.
/// A return of 0 is only expected before the scheduler is fully initialized.
fn monotonic_ns() -> u64 {
    let ret = unsafe { raw_syscall6(SYS_TIME_MONOTONIC, 0, 0, 0, 0, 0, 0) };
    if ret < 0 { 0 } else { ret as u64 }
}

/// ThingOS TimeSpec layout (16 bytes, matches abi::time::TimeSpec).
#[repr(C)]
struct TimeSpec {
    secs: u64,
    nanos: u32,
    _reserved: u32,
}

/// Query SYS_TIME_NOW for the real-time clock; returns total nanoseconds.
fn realtime_ns() -> u64 {
    let mut spec = TimeSpec { secs: 0, nanos: 0, _reserved: 0 };
    let ret = unsafe {
        raw_syscall6(
            SYS_TIME_NOW,
            CLOCK_REALTIME,
            (&mut spec as *mut TimeSpec) as usize,
            0,
            0,
            0,
            0,
        )
    };
    if ret < 0 {
        return 0;
    }
    spec.secs.saturating_mul(1_000_000_000).saturating_add(spec.nanos as u64)
}

// ─── Instant ─────────────────────────────────────────────────────────────────

/// Monotonic instant backed by `SYS_TIME_MONOTONIC`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(Duration);

impl Instant {
    pub fn now() -> Instant {
        Instant(Duration::from_nanos(monotonic_ns()))
    }

    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        self.0.checked_sub(other.0)
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        self.0.checked_add(*other).map(Instant)
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        self.0.checked_sub(*other).map(Instant)
    }
}

// ─── SystemTime ──────────────────────────────────────────────────────────────

/// Wall-clock time backed by `SYS_TIME_NOW` with `ClockId::Realtime`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime(Duration);

pub const UNIX_EPOCH: SystemTime = SystemTime(Duration::from_secs(0));

impl SystemTime {
    pub const MAX: SystemTime = SystemTime(Duration::MAX);
    pub const MIN: SystemTime = SystemTime(Duration::ZERO);

    pub fn now() -> SystemTime {
        SystemTime(Duration::from_nanos(realtime_ns()))
    }

    pub fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        self.0.checked_sub(other.0).ok_or_else(|| other.0 - self.0)
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
        self.0.checked_add(*other).map(SystemTime)
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
        self.0.checked_sub(*other).map(SystemTime)
    }

    /// Construct a `SystemTime` from seconds and nanoseconds since the Unix epoch.
    ///
    /// Used by the filesystem PAL layer to convert VFS stat timestamps into
    /// `std::time::SystemTime` values.
    pub(crate) fn from_timespec(secs: u64, nsecs: u32) -> SystemTime {
        SystemTime(Duration::new(secs, nsecs))
    }

    /// Return the inner `Duration` since the Unix epoch.
    pub(crate) fn as_duration(&self) -> Duration {
        self.0
    }
}
