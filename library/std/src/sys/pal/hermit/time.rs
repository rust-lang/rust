#![allow(dead_code)]

use core::hash::Hash;

use super::hermit_abi::{self, CLOCK_MONOTONIC, CLOCK_REALTIME, timespec};
use crate::ops::{Add, AddAssign, Sub, SubAssign};
use crate::sys::common::timespec::Timespec;
use crate::time::Duration;

const NSEC_PER_SEC: i32 = 1_000_000_000;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(Timespec);

impl Instant {
    pub fn now() -> Instant {
        let mut time: timespec = timespec { tv_sec: 0, tv_nsec: 0 };
        let _ = unsafe { hermit_abi::clock_gettime(CLOCK_MONOTONIC, &raw mut time) };

        Instant(Timespec::new(time.tv_sec, time.tv_nsec as i64).unwrap())
    }

    #[stable(feature = "time2", since = "1.8.0")]
    pub fn elapsed(&self) -> Duration {
        Instant::now() - *self
    }

    pub fn duration_since(&self, earlier: Instant) -> Duration {
        self.checked_duration_since(earlier).unwrap_or_default()
    }

    pub fn checked_duration_since(&self, earlier: Instant) -> Option<Duration> {
        self.checked_sub_instant(&earlier)
    }

    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        self.0.sub_timespec(&other.0).ok()
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant(self.0.checked_add_duration(other)?))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant(self.0.checked_sub_duration(other)?))
    }

    pub fn checked_add(&self, duration: Duration) -> Option<Instant> {
        self.0.checked_add_duration(&duration).map(Instant)
    }

    pub fn checked_sub(&self, duration: Duration) -> Option<Instant> {
        self.0.checked_sub_duration(&duration).map(Instant)
    }
}

impl Add<Duration> for Instant {
    type Output = Instant;

    /// # Panics
    ///
    /// This function may panic if the resulting point in time cannot be represented by the
    /// underlying data structure. See [`Instant::checked_add`] for a version without panic.
    fn add(self, other: Duration) -> Instant {
        self.checked_add(other).expect("overflow when adding duration to instant")
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
        self.checked_sub(other).expect("overflow when subtracting duration from instant")
    }
}

impl SubAssign<Duration> for Instant {
    fn sub_assign(&mut self, other: Duration) {
        *self = *self - other;
    }
}

impl Sub<Instant> for Instant {
    type Output = Duration;

    /// Returns the amount of time elapsed from another instant to this one,
    /// or zero duration if that instant is later than this one.
    ///
    /// # Panics
    ///
    /// Previous Rust versions panicked when `other` was later than `self`. Currently this
    /// method saturates. Future versions may reintroduce the panic in some circumstances.
    /// See [Monotonicity].
    ///
    /// [Monotonicity]: Instant#monotonicity
    fn sub(self, other: Instant) -> Duration {
        self.duration_since(other)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SystemTime(Timespec);

pub const UNIX_EPOCH: SystemTime = SystemTime(Timespec::zero());

impl SystemTime {
    pub const MAX: SystemTime = SystemTime(Timespec::MAX);

    pub const MIN: SystemTime = SystemTime(Timespec::MIN);

    pub fn new(tv_sec: i64, tv_nsec: i32) -> SystemTime {
        SystemTime(Timespec::new(tv_sec, tv_nsec as i64).unwrap())
    }

    pub fn now() -> SystemTime {
        let mut time: timespec = timespec { tv_sec: 0, tv_nsec: 0 };
        let _ = unsafe { hermit_abi::clock_gettime(CLOCK_REALTIME, &raw mut time) };
        SystemTime::new(time.tv_sec, time.tv_nsec)
    }

    pub fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        self.0.sub_timespec(&other.0)
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_add_duration(other)?))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_sub_duration(other)?))
    }
}
