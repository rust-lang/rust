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

    #[rustc_const_unstable(feature = "const_system_time", issue = "144517")]
    pub const fn from_wasi_timestamp(ts: wasi::Timestamp) -> SystemTime {
        SystemTime(Duration::from_nanos(ts))
    }

    #[rustc_const_unstable(feature = "const_system_time", issue = "144517")]
    pub const fn to_wasi_timestamp(&self) -> Option<wasi::Timestamp> {
        // FIXME: const TryInto
        let ns = self.0.as_nanos();
        if ns <= u64::MAX as u128 { Some(ns as u64) } else { None }
    }

    #[rustc_const_unstable(feature = "const_system_time", issue = "144517")]
    pub const fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        // FIXME: ok_or_else with const closures
        match self.0.checked_sub(other.0) {
            Some(duration) => Ok(duration),
            None => Err(other.0 - self.0),
        }
    }

    #[rustc_const_unstable(feature = "const_system_time", issue = "144517")]
    pub const fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_add(*other)?))
    }

    #[rustc_const_unstable(feature = "const_system_time", issue = "144517")]
    pub const fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_sub(*other)?))
    }
}
