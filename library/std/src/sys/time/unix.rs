use crate::sys::AsInner;
use crate::sys::pal::time::Timespec;
use crate::time::Duration;
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

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instant {
    t: Timespec,
}

impl Instant {
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

    /// Returns `self` converted into units of `mach_absolute_time`, or `None`
    /// if `self` is before the system boot time. If the conversion cannot be
    /// performed precisely, this ceils the result up to the nearest
    /// representable value.
    #[cfg(target_vendor = "apple")]
    pub fn into_mach_absolute_time_ceil(self) -> Option<u128> {
        #[repr(C)]
        struct mach_timebase_info {
            numer: u32,
            denom: u32,
        }

        unsafe extern "C" {
            unsafe fn mach_timebase_info(info: *mut mach_timebase_info) -> libc::kern_return_t;
        }

        let secs = u64::try_from(self.t.tv_sec).ok()?;

        let mut timebase = mach_timebase_info { numer: 0, denom: 0 };
        assert_eq!(unsafe { mach_timebase_info(&mut timebase) }, libc::KERN_SUCCESS);

        // Since `tv_sec` is 64-bit and `tv_nsec` is smaller than 1 billion,
        // this cannot overflow. The resulting number needs at most 94 bits.
        let nanos = 1_000_000_000 * u128::from(secs) + u128::from(self.t.tv_nsec.as_inner());
        // This multiplication cannot overflow since multiplying a 94-bit
        // number by a 32-bit number yields a number that needs at most
        // 126 bits.
        Some((nanos * u128::from(timebase.denom)).div_ceil(u128::from(timebase.numer)))
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
