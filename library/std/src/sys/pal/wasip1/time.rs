#![forbid(unsafe_op_in_unsafe_fn)]

use crate::time::Duration;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant {
    nanos: wasi::Timestamp,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime {
    nanos: wasi::Timestamp,
}

pub const UNIX_EPOCH: SystemTime = SystemTime { nanos: 0 };

fn current_time(clock: wasi::Clockid) -> wasi::Timestamp {
    unsafe {
        wasi::clock_time_get(
            clock, 1, // precision... seems ignored though?
        )
        .unwrap()
    }
}

impl Instant {
    pub fn now() -> Instant {
        Instant { nanos: current_time(wasi::CLOCKID_MONOTONIC) }
    }

    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        let nanos = self.nanos.checked_sub(other.nanos)?;
        Some(Duration::from_nanos(nanos))
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        let to_add = other.as_nanos().try_into().ok()?;
        let nanos = self.nanos.checked_add(to_add)?;
        Some(Instant { nanos })
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        let to_sub = other.as_nanos().try_into().ok()?;
        let nanos = self.nanos.checked_sub(to_sub)?;
        Some(Instant { nanos })
    }
}

impl SystemTime {
    pub fn now() -> SystemTime {
        SystemTime { nanos: current_time(wasi::CLOCKID_REALTIME) }
    }

    pub fn from_wasi_timestamp(ts: wasi::Timestamp) -> SystemTime {
        SystemTime { nanos: ts }
    }

    pub fn to_wasi_timestamp(self) -> wasi::Timestamp {
        self.nanos
    }

    pub fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        self.nanos
            .checked_sub(other.nanos)
            .map(Duration::from_nanos)
            .ok_or_else(|| Duration::from_nanos(other.nanos - self.nanos))
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
        let to_add = other.as_nanos().try_into().ok()?;
        let nanos = self.nanos.checked_add(to_add)?;
        Some(SystemTime { nanos })
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
        let to_sub = other.as_nanos().try_into().ok()?;
        let nanos = self.nanos.checked_sub(to_sub)?;
        Some(SystemTime { nanos })
    }
}
