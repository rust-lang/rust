use crate::mem;
use crate::sys::cloudabi::abi;
use crate::time::Duration;

const NSEC_PER_SEC: abi::timestamp = 1_000_000_000;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant {
    t: abi::timestamp,
}

pub fn checked_dur2intervals(dur: &Duration) -> Option<abi::timestamp> {
    dur.as_secs()
        .checked_mul(NSEC_PER_SEC)?
        .checked_add(dur.subsec_nanos() as abi::timestamp)
}

impl Instant {
    pub fn now() -> Instant {
        unsafe {
            let mut t = mem::uninitialized();
            let ret = abi::clock_time_get(abi::clockid::MONOTONIC, 0, &mut t);
            assert_eq!(ret, abi::errno::SUCCESS);
            Instant { t }
        }
    }

    pub fn actually_monotonic() -> bool {
        true
    }

    pub const fn zero() -> Instant {
        Instant { t: 0 }
    }

    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        let diff = self.t.checked_sub(other.t)?;
        Some(Duration::new(diff / NSEC_PER_SEC, (diff % NSEC_PER_SEC) as u32))
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant {
            t: self.t.checked_add(checked_dur2intervals(other)?)?,
        })
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant {
            t: self.t.checked_sub(checked_dur2intervals(other)?)?,
        })
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime {
    t: abi::timestamp,
}

impl SystemTime {
    pub fn now() -> SystemTime {
        unsafe {
            let mut t = mem::uninitialized();
            let ret = abi::clock_time_get(abi::clockid::REALTIME, 0, &mut t);
            assert_eq!(ret, abi::errno::SUCCESS);
            SystemTime { t }
        }
    }

    pub fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        if self.t >= other.t {
            let diff = self.t - other.t;
            Ok(Duration::new(
                diff / NSEC_PER_SEC,
                (diff % NSEC_PER_SEC) as u32,
            ))
        } else {
            let diff = other.t - self.t;
            Err(Duration::new(
                diff / NSEC_PER_SEC,
                (diff % NSEC_PER_SEC) as u32,
            ))
        }
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime {
            t: self.t.checked_add(checked_dur2intervals(other)?)?,
        })
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime {
            t: self.t.checked_sub(checked_dur2intervals(other)?)?,
        })
    }
}

pub const UNIX_EPOCH: SystemTime = SystemTime { t: 0 };
