use time::Duration;
use sys::{TimeSysCall, TimeClock};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(Duration);

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime(Duration);

pub const UNIX_EPOCH: SystemTime = SystemTime(Duration::from_secs(0));

impl Instant {
    pub fn now() -> Instant {
        Instant(TimeSysCall::perform(TimeClock::Monotonic))
    }

    pub fn sub_instant(&self, other: &Instant) -> Duration {
        self.0 - other.0
    }

    pub fn add_duration(&self, other: &Duration) -> Instant {
        Instant(self.0 + *other)
    }

    pub fn sub_duration(&self, other: &Duration) -> Instant {
        Instant(self.0 - *other)
    }
}

impl SystemTime {
    pub fn now() -> SystemTime {
        SystemTime(TimeSysCall::perform(TimeClock::System))
    }

    pub fn sub_time(&self, other: &SystemTime)
                    -> Result<Duration, Duration> {
        self.0.checked_sub(other.0).ok_or_else(|| other.0 - self.0)
    }

    pub fn add_duration(&self, other: &Duration) -> SystemTime {
        SystemTime(self.0 + *other)
    }

    pub fn sub_duration(&self, other: &Duration) -> SystemTime {
        SystemTime(self.0 - *other)
    }
}
