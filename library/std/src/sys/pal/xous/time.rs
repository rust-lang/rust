use crate::os::xous::ffi::blocking_scalar;
use crate::os::xous::services::SystimeScalar::GetUtcTimeMs;
use crate::os::xous::services::TicktimerScalar::ElapsedMs;
use crate::os::xous::services::{systime_server, ticktimer_server};
use crate::time::Duration;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant {
    millis: u64,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime {
    millis: u64,
}

pub const UNIX_EPOCH: SystemTime = SystemTime { millis: 0 };

impl Instant {
    pub fn now() -> Instant {
        let result = blocking_scalar(ticktimer_server(), ElapsedMs.into())
            .expect("failed to request elapsed_ms");
        let lower = result[0];
        let upper = result[1];
        Instant { millis: lower as u64 | (upper as u64) << 32 }
    }

    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        let millis = self.millis.checked_sub(other.millis)?;
        Some(Duration::from_millis(millis))
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        let to_add = other.as_millis().try_into().ok()?;
        let millis = self.millis.checked_add(to_add)?;
        Some(Instant { millis })
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        let to_sub = other.as_millis().try_into().ok()?;
        let millis = self.millis.checked_sub(to_sub)?;
        Some(Instant { millis })
    }
}

impl SystemTime {
    pub fn now() -> SystemTime {
        let result = blocking_scalar(systime_server(), GetUtcTimeMs.into())
            .expect("failed to request utc time in ms");
        let lower = result[0];
        let upper = result[1];
        SystemTime { millis: (upper as u64) << 32 | lower as u64 }
    }

    pub fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        self.millis
            .checked_sub(other.millis)
            .map(Duration::from_millis)
            .ok_or_else(|| Duration::from_millis(other.millis - self.millis))
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
        let to_add = other.as_millis().try_into().ok()?;
        let millis = self.millis.checked_add(to_add)?;
        Some(SystemTime { millis })
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
        let to_sub = other.as_millis().try_into().ok()?;
        let millis = self.millis.checked_sub(to_sub)?;
        Some(SystemTime { millis })
    }
}
