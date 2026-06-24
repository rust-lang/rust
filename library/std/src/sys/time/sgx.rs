use core::num::niche_types::Nanoseconds;

use crate::sys::pal::abi::usercalls;
use crate::time::{Duration, Instant};

pub fn now() -> Instant {
    let time = usercalls::insecure_time();
    let secs = (time / 1_000_000_000) as i64;
    let nanos = Nanoseconds::new((time % 1_000_000_000) as u32).unwrap();
    Instant { secs, nanos }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime(Duration);

pub const UNIX_EPOCH: SystemTime = SystemTime(Duration::from_secs(0));

impl SystemTime {
    pub const MAX: SystemTime = SystemTime(Duration::MAX);

    pub const MIN: SystemTime = SystemTime(Duration::ZERO);

    pub fn now() -> SystemTime {
        let t = usercalls::insecure_time();
        SystemTime(Duration::new(t / 1_000_000_000, (t % 1_000_000_000) as _))
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
