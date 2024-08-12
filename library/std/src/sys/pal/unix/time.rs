use crate::time::Duration;
use crate::{fmt, io};

mod timespec;
pub use timespec::*;

pub const UNIX_EPOCH: SystemTime = SystemTime { t: Timespec::zero() };
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SystemTime {
    pub(crate) t: Timespec,
}

impl SystemTime {
    #[cfg_attr(any(target_os = "horizon", target_os = "hurd"), allow(unused))]
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
            .field("tv_nsec", &self.t.tv_nsec.0)
            .finish()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instant {
    t: Timespec,
}

impl Instant {
    pub fn now() -> Instant {
        // https://www.manpagez.com/man/3/clock_gettime/
        //
        // CLOCK_UPTIME_RAW   clock that increments monotonically, in the same man-
        //                    ner as CLOCK_MONOTONIC_RAW, but that does not incre-
        //                    ment while the system is asleep.  The returned value
        //                    is identical to the result of mach_absolute_time()
        //                    after the appropriate mach_timebase conversion is
        //                    applied.
        //
        // Instant on macos was historically implemented using mach_absolute_time;
        // we preserve this value domain out of an abundance of caution.
        #[cfg(target_vendor = "apple")]
        const clock_id: libc::clockid_t = libc::CLOCK_UPTIME_RAW;
        #[cfg(not(target_vendor = "apple"))]
        const clock_id: libc::clockid_t = libc::CLOCK_MONOTONIC;
        Instant { t: Timespec::now(clock_id) }
    }

    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        self.t.sub_timespec(&other.t).ok()
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant { t: self.t.checked_add_duration(other)? })
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant { t: self.t.checked_sub_duration(other)? })
    }
}

impl fmt::Debug for Instant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Instant")
            .field("tv_sec", &self.t.tv_sec)
            .field("tv_nsec", &self.t.tv_nsec.0)
            .finish()
    }
}
