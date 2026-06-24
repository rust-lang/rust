use hermit_abi::{self, CLOCK_MONOTONIC, CLOCK_REALTIME};

use crate::io;
use crate::sys::cvt;
use crate::sys::pal::time::Timespec;
use crate::time::{Duration, Instant};

fn clock_gettime(clock: hermit_abi::clockid_t) -> Timespec {
    let mut t = hermit_abi::timespec { tv_sec: 0, tv_nsec: 0 };
    cvt(unsafe { hermit_abi::clock_gettime(clock, &raw mut t) }).unwrap();
    Timespec::new(t.tv_sec, t.tv_nsec.into()).unwrap()
}

pub fn now() -> Instant {
    let time = clock_gettime(CLOCK_MONOTONIC);
    Instant { secs: time.tv_sec, nanos: time.tv_nsec }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SystemTime(Timespec);

pub const UNIX_EPOCH: SystemTime = SystemTime(Timespec::zero());

impl SystemTime {
    pub const MAX: SystemTime = SystemTime(Timespec::MAX);

    pub const MIN: SystemTime = SystemTime(Timespec::MIN);

    pub fn new(tv_sec: i64, tv_nsec: i64) -> Result<SystemTime, io::Error> {
        Ok(SystemTime(Timespec::new(tv_sec, tv_nsec)?))
    }

    pub fn now() -> SystemTime {
        SystemTime(clock_gettime(CLOCK_REALTIME))
    }

    pub fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        self.0.sub_timespec(&other.0)
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_add_duration(other)?))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_sub_duration(other)?))
    }
}
