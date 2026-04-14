//! ThingOS time implementation.
//!
//! `Instant` wraps `CLOCK_MONOTONIC` and `SystemTime` wraps `CLOCK_REALTIME`,
//! both obtained via `SYS_CLOCK_GETTIME`.

#![allow(dead_code)]

use core::hash::{Hash, Hasher};

use super::common::{CLOCK_MONOTONIC, CLOCK_REALTIME, SYS_CLOCK_GETTIME, Timespec, syscall2};
use crate::cmp::Ordering;
use crate::ops::{Add, AddAssign, Sub, SubAssign};
use crate::time::Duration;

const NSEC_PER_SEC: i64 = 1_000_000_000;

fn clock_gettime(clk_id: u64) -> Timespec {
    let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe {
        syscall2(SYS_CLOCK_GETTIME, clk_id, &raw mut ts as u64);
    }
    ts
}

// ── Internal Timespec helpers ────────────────────────────────────────────────

#[derive(Copy, Clone, Debug)]
struct Ts(Timespec);

impl Ts {
    const ZERO: Ts = Ts(Timespec { tv_sec: 0, tv_nsec: 0 });
    const MAX: Ts = Ts(Timespec { tv_sec: i64::MAX, tv_nsec: NSEC_PER_SEC - 1 });
    const MIN: Ts = Ts(Timespec { tv_sec: i64::MIN, tv_nsec: 0 });

    fn sub_ts(&self, other: &Ts) -> Result<Duration, Duration> {
        if self >= other {
            let (sec_diff, nsec_diff) = if self.0.tv_nsec >= other.0.tv_nsec {
                (
                    (self.0.tv_sec - other.0.tv_sec) as u64,
                    (self.0.tv_nsec - other.0.tv_nsec) as u32,
                )
            } else {
                (
                    (self.0.tv_sec - 1 - other.0.tv_sec) as u64,
                    (self.0.tv_nsec + NSEC_PER_SEC - other.0.tv_nsec) as u32,
                )
            };
            Ok(Duration::new(sec_diff, nsec_diff))
        } else {
            match other.sub_ts(self) {
                Ok(d) => Err(d),
                Err(d) => Ok(d),
            }
        }
    }

    fn checked_add_duration(&self, d: &Duration) -> Option<Ts> {
        let mut secs = self.0.tv_sec.checked_add_unsigned(d.as_secs())?;
        let mut nsec = self.0.tv_nsec + d.subsec_nanos() as i64;
        if nsec >= NSEC_PER_SEC {
            nsec -= NSEC_PER_SEC;
            secs = secs.checked_add(1)?;
        }
        Some(Ts(Timespec { tv_sec: secs, tv_nsec: nsec }))
    }

    fn checked_sub_duration(&self, d: &Duration) -> Option<Ts> {
        let mut secs = self.0.tv_sec.checked_sub_unsigned(d.as_secs())?;
        let mut nsec = self.0.tv_nsec - d.subsec_nanos() as i64;
        if nsec < 0 {
            nsec += NSEC_PER_SEC;
            secs = secs.checked_sub(1)?;
        }
        Some(Ts(Timespec { tv_sec: secs, tv_nsec: nsec }))
    }
}

impl PartialEq for Ts {
    fn eq(&self, other: &Ts) -> bool {
        self.0.tv_sec == other.0.tv_sec && self.0.tv_nsec == other.0.tv_nsec
    }
}

impl Eq for Ts {}

impl PartialOrd for Ts {
    fn partial_cmp(&self, other: &Ts) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ts {
    fn cmp(&self, other: &Ts) -> Ordering {
        (self.0.tv_sec, self.0.tv_nsec).cmp(&(other.0.tv_sec, other.0.tv_nsec))
    }
}

impl Hash for Ts {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.tv_sec.hash(state);
        self.0.tv_nsec.hash(state);
    }
}

// ── Instant ──────────────────────────────────────────────────────────────────

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(Ts);

impl Instant {
    pub fn now() -> Instant {
        Instant(Ts(clock_gettime(CLOCK_MONOTONIC)))
    }

    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        self.0.sub_ts(&other.0).ok()
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant(self.0.checked_add_duration(other)?))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant(self.0.checked_sub_duration(other)?))
    }
}

impl Add<Duration> for Instant {
    type Output = Instant;
    fn add(self, other: Duration) -> Instant {
        self.checked_add_duration(&other).expect("overflow when adding duration to instant")
    }
}

impl AddAssign<Duration> for Instant {
    fn add_assign(&mut self, other: Duration) {
        *self = *self + other;
    }
}

impl Sub<Duration> for Instant {
    type Output = Instant;
    fn sub(self, other: Duration) -> Instant {
        self.checked_sub_duration(&other)
            .expect("overflow when subtracting duration from instant")
    }
}

impl SubAssign<Duration> for Instant {
    fn sub_assign(&mut self, other: Duration) {
        *self = *self - other;
    }
}

impl Sub<Instant> for Instant {
    type Output = Duration;
    fn sub(self, other: Instant) -> Duration {
        self.checked_sub_instant(&other).unwrap_or_default()
    }
}

// ── SystemTime ───────────────────────────────────────────────────────────────

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SystemTime(Ts);

pub const UNIX_EPOCH: SystemTime = SystemTime(Ts::ZERO);

impl SystemTime {
    pub const MAX: SystemTime = SystemTime(Ts::MAX);
    pub const MIN: SystemTime = SystemTime(Ts::MIN);

    pub fn new(tv_sec: i64, tv_nsec: i64) -> SystemTime {
        SystemTime(Ts(Timespec { tv_sec, tv_nsec }))
    }

    pub fn now() -> SystemTime {
        SystemTime(Ts(clock_gettime(CLOCK_REALTIME)))
    }

    pub fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        self.0.sub_ts(&other.0)
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_add_duration(other)?))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_sub_duration(other)?))
    }

    /// Convenience: encode as nanoseconds since UNIX epoch (used by fs).
    pub fn as_nanos_since_epoch(&self) -> u128 {
        let ts = &self.0.0;
        if ts.tv_sec < 0 {
            return 0;
        }
        ts.tv_sec as u128 * 1_000_000_000u128 + ts.tv_nsec as u128
    }

    /// Convenience: decode from nanoseconds since UNIX epoch (used by fs).
    pub fn from_nanos_since_epoch(ns: u128) -> SystemTime {
        let secs = (ns / 1_000_000_000) as i64;
        let nsec = (ns % 1_000_000_000) as i64;
        SystemTime(Ts(Timespec { tv_sec: secs, tv_nsec: nsec }))
    }
}
