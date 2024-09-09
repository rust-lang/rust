use crate::os::xous::ffi::blocking_scalar;
use crate::os::xous::services::SystimeScalar::GetUtcTimeMs;
use crate::os::xous::services::TicktimerScalar::ElapsedMs;
use crate::os::xous::services::{systime_server, ticktimer_server};
use crate::time::Duration;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(Duration);

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime(Duration);

pub const UNIX_EPOCH: SystemTime = SystemTime(Duration::from_secs(0));

impl Instant {
    pub fn now() -> Instant {
        let result = blocking_scalar(ticktimer_server(), ElapsedMs.into())
            .expect("failed to request elapsed_ms");
        let lower = result[0];
        let upper = result[1];
        Instant { 0: Duration::from_millis(lower as u64 | (upper as u64) << 32) }
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

impl SystemTime {
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
