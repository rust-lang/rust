use crate::time::Duration;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(Duration);

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime(Duration);

pub const UNIX_EPOCH: SystemTime = SystemTime(Duration::from_secs(0));

impl Instant {
    pub fn now() -> Instant {
        Instant(Duration::from_nanos(wasip2::clocks::monotonic_clock::now()))
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

    pub(crate) fn as_duration(&self) -> &Duration {
        &self.0
    }
}

impl SystemTime {
    pub fn now() -> SystemTime {
        let now = wasip2::clocks::wall_clock::now();
        SystemTime(Duration::new(now.seconds, now.nanoseconds))
    }

    pub fn from_wasi_timestamp(ts: wasi::Timestamp) -> SystemTime {
        SystemTime(Duration::from_nanos(ts))
    }

    pub fn to_wasi_timestamp(&self) -> Option<wasi::Timestamp> {
        self.0.as_nanos().try_into().ok()
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
