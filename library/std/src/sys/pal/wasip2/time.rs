use crate::time::Duration;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant {
    nanos: wasip2::clocks::monotonic_clock::Instant,
}

// WASIp2's datetime is identical to our `Duration` in terms of its representable
// range, so use `Duration` to simplify the implementation below.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime(Duration);

pub const UNIX_EPOCH: SystemTime = SystemTime(Duration::from_secs(0));

impl Instant {
    pub fn now() -> Instant {
        Instant { nanos: wasip2::clocks::monotonic_clock::now() }
    }

    pub fn to_wasi_instant(self) -> wasip2::clocks::monotonic_clock::Instant {
        self.nanos
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
        let now = wasip2::clocks::wall_clock::now();
        SystemTime(Duration::new(now.seconds, now.nanoseconds))
    }

    pub fn from_wasi_timestamp(ts: wasi::Timestamp) -> SystemTime {
        SystemTime(Duration::from_nanos(ts))
    }

    pub fn to_wasi_timestamp(self) -> wasi::Timestamp {
        // FIXME: use the WASIp2 filesystem proposal, which accepts a WASIp2 datetime.
        self.0.as_nanos().try_into().expect("error converting WASIp2 datetime to WASIp1 timestamp")
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
