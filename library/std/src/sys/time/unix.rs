use crate::sys::pal::time::Timespec;
use crate::time::{Duration, Instant};
use crate::{fmt, io};

pub const UNIX_EPOCH: SystemTime = SystemTime { t: Timespec::zero() };

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SystemTime {
    pub(crate) t: Timespec,
}

impl SystemTime {
    pub const MAX: SystemTime = SystemTime { t: Timespec::MAX };

    pub const MIN: SystemTime = SystemTime { t: Timespec::MIN };

    #[cfg_attr(any(target_os = "horizon", target_os = "hurd", target_os = "teeos"), expect(unused))]
    pub fn new(tv_sec: i64, tv_nsec: i64) -> Result<SystemTime, io::Error> {
        Ok(SystemTime { t: Timespec::new(tv_sec, tv_nsec)? })
    }

    pub fn now() -> SystemTime {
        SystemTime { t: Timespec::now(libc::CLOCK_REALTIME) }
    }

    pub fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        self.t.sub_timespec(&other.t)
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime { t: self.t.checked_add_duration(other)? })
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime { t: self.t.checked_sub_duration(other)? })
    }
}

impl fmt::Debug for SystemTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SystemTime")
            .field("tv_sec", &self.t.tv_sec)
            .field("tv_nsec", &self.t.tv_nsec)
            .finish()
    }
}

// CLOCK_UPTIME_RAW   clock that increments monotonically, in the same man-
//                    ner as CLOCK_MONOTONIC_RAW, but that does not incre-
//                    ment while the system is asleep.  The returned value
//                    is identical to the result of mach_absolute_time()
//                    after the appropriate mach_timebase conversion is
//                    applied.
//
// We use `CLOCK_UPTIME_RAW` instead of `CLOCK_MONOTONIC` since
// `CLOCK_UPTIME_RAW` is based on `mach_absolute_time`, which is the
// clock that all timeouts and deadlines are measured against inside
// the kernel.
#[cfg(target_vendor = "apple")]
pub(crate) const CLOCK_ID: libc::clockid_t = libc::CLOCK_UPTIME_RAW;
#[cfg(not(target_vendor = "apple"))]
pub(crate) const CLOCK_ID: libc::clockid_t = libc::CLOCK_MONOTONIC;

pub fn now() -> Instant {
    // https://pubs.opengroup.org/onlinepubs/9799919799/functions/clock_getres.html
    let time = Timespec::now(CLOCK_ID);
    Instant { secs: time.tv_sec, nanos: time.tv_nsec }
}
