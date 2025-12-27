use crate::sys::AsInner;
use crate::sys::common::timespec::Timespec;
use crate::time::Duration;
use crate::{fmt, io};

pub const UNIX_EPOCH: SystemTime = SystemTime { t: Timespec::zero() };
#[allow(dead_code)] // Used for pthread condvar timeouts
pub const TIMESPEC_MAX: libc::timespec =
    libc::timespec { tv_sec: <libc::time_t>::MAX, tv_nsec: 1_000_000_000 - 1 };

// This additional constant is only used when calling
// `libc::pthread_cond_timedwait`.
#[cfg(target_os = "nto")]
pub(in crate::sys) const TIMESPEC_MAX_CAPPED: libc::timespec = libc::timespec {
    tv_sec: (u64::MAX / NSEC_PER_SEC) as i64,
    tv_nsec: (u64::MAX % NSEC_PER_SEC) as i64,
};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SystemTime {
    pub(crate) t: Timespec,
}

impl SystemTime {
    pub const MAX: SystemTime = SystemTime { t: Timespec::MAX };

    pub const MIN: SystemTime = SystemTime { t: Timespec::MIN };

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
            .field("tv_nsec", &self.t.tv_nsec)
            .finish()
    }
}

#[cfg(all(
    target_os = "linux",
    target_env = "gnu",
    target_pointer_width = "32",
    not(target_arch = "riscv32")
))]
#[repr(C)]
pub(crate) struct __timespec64 {
    pub(crate) tv_sec: i64,
    #[cfg(target_endian = "big")]
    _padding: i32,
    pub(crate) tv_nsec: i32,
    #[cfg(target_endian = "little")]
    _padding: i32,
}

#[cfg(all(
    target_os = "linux",
    target_env = "gnu",
    target_pointer_width = "32",
    not(target_arch = "riscv32")
))]
impl __timespec64 {
    pub(crate) fn new(tv_sec: i64, tv_nsec: i32) -> Self {
        Self { tv_sec, tv_nsec, _padding: 0 }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instant {
    t: Timespec,
}

impl Instant {
    #[cfg(target_vendor = "apple")]
    pub(crate) const CLOCK_ID: libc::clockid_t = libc::CLOCK_UPTIME_RAW;
    #[cfg(not(target_vendor = "apple"))]
    pub(crate) const CLOCK_ID: libc::clockid_t = libc::CLOCK_MONOTONIC;
    pub fn now() -> Instant {
        // https://pubs.opengroup.org/onlinepubs/9799919799/functions/clock_getres.html
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
        Instant { t: Timespec::now(Self::CLOCK_ID) }
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

    #[cfg_attr(
        not(target_os = "linux"),
        allow(unused, reason = "needed by the `sleep_until` on some unix platforms")
    )]
    pub(crate) fn into_timespec(self) -> Timespec {
        self.t
    }
}

impl AsInner<Timespec> for Instant {
    fn as_inner(&self) -> &Timespec {
        &self.t
    }
}

impl fmt::Debug for Instant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Instant")
            .field("tv_sec", &self.t.tv_sec)
            .field("tv_nsec", &self.t.tv_nsec)
            .finish()
    }
}
