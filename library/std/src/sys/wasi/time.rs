#![deny(unsafe_op_in_unsafe_fn)]

use crate::time::Duration;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(Duration);

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime(Duration);

pub const UNIX_EPOCH: SystemTime = SystemTime(Duration::from_secs(0));

fn current_time(clock: wasi::Clockid) -> Duration {
    let ts = unsafe {
        wasi::clock_time_get(
            clock, 1, // precision... seems ignored though?
        )
        .unwrap()
    };
    Duration::new((ts / 1_000_000_000) as u64, (ts % 1_000_000_000) as u32)
}

impl Instant {
    pub fn now() -> Instant {
        Instant(current_time(wasi::CLOCKID_MONOTONIC))
    }

    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        self.0.checked_sub(other.0)
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant(self.0.checked_add(*other)?))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant(self.0.checked_sub(*other)?))
    }
}

impl SystemTime {
    pub fn now() -> SystemTime {
        SystemTime(current_time(wasi::CLOCKID_REALTIME))
    }

    pub fn from_wasi_timestamp(ts: wasi::Timestamp) -> SystemTime {
        SystemTime(Duration::from_nanos(ts))
    }

    pub fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        self.0.checked_sub(other.0).ok_or_else(|| other.0 - self.0)
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_add(*other)?))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_sub(*other)?))
    }
}
