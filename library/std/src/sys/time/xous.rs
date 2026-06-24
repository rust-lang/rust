use core::num::niche_types::Nanoseconds;

use crate::os::xous::ffi::blocking_scalar;
use crate::os::xous::services::SystimeScalar::GetUtcTimeMs;
use crate::os::xous::services::TicktimerScalar::ElapsedMs;
use crate::os::xous::services::{systime_server, ticktimer_server};
use crate::time::{Duration, Instant};

pub fn now() -> Instant {
    let result = blocking_scalar(ticktimer_server(), ElapsedMs.into())
        .expect("failed to request elapsed_ms");
    let lower = result[0];
    let upper = result[1];
    let millis = lower as u64 | (upper as u64) << 32;

    let secs = (millis / 1_000) as i64;
    let nanos = Nanoseconds::new(1_000_000 * (millis % 1000) as u32).unwrap();
    Instant { secs, nanos }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime(Duration);

pub const UNIX_EPOCH: SystemTime = SystemTime(Duration::from_secs(0));

impl SystemTime {
    pub const MAX: SystemTime = SystemTime(Duration::MAX);

    pub const MIN: SystemTime = SystemTime(Duration::ZERO);

    pub fn now() -> SystemTime {
        let result = blocking_scalar(systime_server(), GetUtcTimeMs.into())
            .expect("failed to request utc time in ms");
        let lower = result[0];
        let upper = result[1];
        SystemTime { 0: Duration::from_millis((upper as u64) << 32 | lower as u64) }
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
