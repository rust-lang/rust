use super::abi::usercalls;
use crate::time::Duration;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant {
    nanos: u64,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime {
    nanos: u64,
}

pub const UNIX_EPOCH: SystemTime = SystemTime { nanos: 0 };

impl Instant {
    pub fn now() -> Instant {
        Instant { nanos: usercalls::insecure_time() }
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
        SystemTime { nanos: usercalls::insecure_time() }
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
