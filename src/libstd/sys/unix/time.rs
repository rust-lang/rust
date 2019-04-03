use crate::cmp::Ordering;
use crate::time::Duration;

use core::hash::{Hash, Hasher};

pub use self::inner::{Instant, SystemTime, UNIX_EPOCH};
use crate::convert::TryInto;

const NSEC_PER_SEC: u64 = 1_000_000_000;

#[derive(Copy, Clone)]
struct Timespec {
    t: libc::timespec,
}

impl Timespec {
    const fn zero() -> Timespec {
        Timespec {
            t: libc::timespec { tv_sec: 0, tv_nsec: 0 },
        }
    }

    fn sub_timespec(&self, other: &Timespec) -> Result<Duration, Duration> {
        if self >= other {
            Ok(if self.t.tv_nsec >= other.t.tv_nsec {
                Duration::new((self.t.tv_sec - other.t.tv_sec) as u64,
                              (self.t.tv_nsec - other.t.tv_nsec) as u32)
            } else {
                Duration::new((self.t.tv_sec - 1 - other.t.tv_sec) as u64,
                              self.t.tv_nsec as u32 + (NSEC_PER_SEC as u32) -
                              other.t.tv_nsec as u32)
            })
        } else {
            match other.sub_timespec(self) {
                Ok(d) => Err(d),
                Err(d) => Ok(d),
            }
        }
    }

    fn checked_add_duration(&self, other: &Duration) -> Option<Timespec> {
        let mut secs = other
            .as_secs()
            .try_into() // <- target type would be `libc::time_t`
            .ok()
            .and_then(|secs| self.t.tv_sec.checked_add(secs))?;

        // Nano calculations can't overflow because nanos are <1B which fit
        // in a u32.
        let mut nsec = other.subsec_nanos() + self.t.tv_nsec as u32;
        if nsec >= NSEC_PER_SEC as u32 {
            nsec -= NSEC_PER_SEC as u32;
            secs = secs.checked_add(1)?;
        }
        Some(Timespec {
            t: libc::timespec {
                tv_sec: secs,
                tv_nsec: nsec as _,
            },
        })
    }

    fn checked_sub_duration(&self, other: &Duration) -> Option<Timespec> {
        let mut secs = other
            .as_secs()
            .try_into() // <- target type would be `libc::time_t`
            .ok()
            .and_then(|secs| self.t.tv_sec.checked_sub(secs))?;

        // Similar to above, nanos can't overflow.
        let mut nsec = self.t.tv_nsec as i32 - other.subsec_nanos() as i32;
        if nsec < 0 {
            nsec += NSEC_PER_SEC as i32;
            secs = secs.checked_sub(1)?;
        }
        Some(Timespec {
            t: libc::timespec {
                tv_sec: secs,
                tv_nsec: nsec as _,
            },
        })
    }
}

impl PartialEq for Timespec {
    fn eq(&self, other: &Timespec) -> bool {
        self.t.tv_sec == other.t.tv_sec && self.t.tv_nsec == other.t.tv_nsec
    }
}

impl Eq for Timespec {}

impl PartialOrd for Timespec {
    fn partial_cmp(&self, other: &Timespec) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Timespec {
    fn cmp(&self, other: &Timespec) -> Ordering {
        let me = (self.t.tv_sec, self.t.tv_nsec);
        let other = (other.t.tv_sec, other.t.tv_nsec);
        me.cmp(&other)
    }
}

impl Hash for Timespec {
    fn hash<H : Hasher>(&self, state: &mut H) {
        self.t.tv_sec.hash(state);
        self.t.tv_nsec.hash(state);
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod inner {
    use crate::fmt;
    use crate::mem;
    use crate::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use crate::sys::cvt;
    use crate::sys_common::mul_div_u64;
    use crate::time::Duration;

    use super::NSEC_PER_SEC;
    use super::Timespec;

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
    pub struct Instant {
        t: u64
    }

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct SystemTime {
        t: Timespec,
    }

    pub const UNIX_EPOCH: SystemTime = SystemTime {
        t: Timespec::zero(),
    };

    impl Instant {
        pub fn now() -> Instant {
            Instant { t: unsafe { libc::mach_absolute_time() } }
        }

        pub const fn zero() -> Instant {
            Instant { t: 0 }
        }

        pub fn actually_monotonic() -> bool {
            true
        }

        pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
            let diff = self.t.checked_sub(other.t)?;
            let info = info();
            let nanos = mul_div_u64(diff, info.numer as u64, info.denom as u64);
            Some(Duration::new(nanos / NSEC_PER_SEC, (nanos % NSEC_PER_SEC) as u32))
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

    impl SystemTime {
        pub fn now() -> SystemTime {
            use crate::ptr;

            let mut s = libc::timeval {
                tv_sec: 0,
                tv_usec: 0,
            };
            cvt(unsafe {
                libc::gettimeofday(&mut s, ptr::null_mut())
            }).unwrap();
            return SystemTime::from(s)
        }

        pub fn sub_time(&self, other: &SystemTime)
                        -> Result<Duration, Duration> {
            self.t.sub_timespec(&other.t)
        }

        pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
            Some(SystemTime { t: self.t.checked_add_duration(other)? })
        }

        pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
            Some(SystemTime { t: self.t.checked_sub_duration(other)? })
        }
    }

    impl From<libc::timeval> for SystemTime {
        fn from(t: libc::timeval) -> SystemTime {
            SystemTime::from(libc::timespec {
                tv_sec: t.tv_sec,
                tv_nsec: (t.tv_usec * 1000) as libc::c_long,
            })
        }
    }

    impl From<libc::timespec> for SystemTime {
        fn from(t: libc::timespec) -> SystemTime {
            SystemTime { t: Timespec { t } }
        }
    }

    impl fmt::Debug for SystemTime {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("SystemTime")
             .field("tv_sec", &self.t.t.tv_sec)
             .field("tv_nsec", &self.t.t.tv_nsec)
             .finish()
        }
    }

    fn checked_dur2intervals(dur: &Duration) -> Option<u64> {
        let nanos = dur.as_secs()
            .checked_mul(NSEC_PER_SEC)?
            .checked_add(dur.subsec_nanos() as u64)?;
        let info = info();
        Some(mul_div_u64(nanos, info.denom as u64, info.numer as u64))
    }

    fn info() -> libc::mach_timebase_info {
        static mut INFO: libc::mach_timebase_info = libc::mach_timebase_info {
            numer: 0,
            denom: 0,
        };
        static STATE: AtomicUsize = AtomicUsize::new(0);

        unsafe {
            // If a previous thread has filled in this global state, use that.
            if STATE.load(SeqCst) == 2 {
                return INFO;
            }

            // ... otherwise learn for ourselves ...
            let mut info = mem::zeroed();
            libc::mach_timebase_info(&mut info);

            // ... and attempt to be the one thread that stores it globally for
            // all other threads
            if STATE.compare_exchange(0, 1, SeqCst, SeqCst).is_ok() {
                INFO = info;
                STATE.store(2, SeqCst);
            }
            return info;
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
mod inner {
    use crate::fmt;
    use crate::sys::cvt;
    use crate::time::Duration;

    use super::Timespec;

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Instant {
        t: Timespec,
    }

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct SystemTime {
        t: Timespec,
    }

    pub const UNIX_EPOCH: SystemTime = SystemTime {
        t: Timespec::zero(),
    };

    impl Instant {
        pub fn now() -> Instant {
            Instant { t: now(libc::CLOCK_MONOTONIC) }
        }

        pub const fn zero() -> Instant {
            Instant {
                t: Timespec::zero(),
            }
        }

        pub fn actually_monotonic() -> bool {
            (cfg!(target_os = "linux") && cfg!(target_arch = "x86_64")) ||
            (cfg!(target_os = "linux") && cfg!(target_arch = "x86")) ||
            false // last clause, used so `||` is always trailing above
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
             .field("tv_sec", &self.t.t.tv_sec)
             .field("tv_nsec", &self.t.t.tv_nsec)
             .finish()
        }
    }

    impl SystemTime {
        pub fn now() -> SystemTime {
            SystemTime { t: now(libc::CLOCK_REALTIME) }
        }

        pub fn sub_time(&self, other: &SystemTime)
                        -> Result<Duration, Duration> {
            self.t.sub_timespec(&other.t)
        }

        pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
            Some(SystemTime { t: self.t.checked_add_duration(other)? })
        }

        pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
            Some(SystemTime { t: self.t.checked_sub_duration(other)? })
        }
    }

    impl From<libc::timespec> for SystemTime {
        fn from(t: libc::timespec) -> SystemTime {
            SystemTime { t: Timespec { t } }
        }
    }

    impl fmt::Debug for SystemTime {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("SystemTime")
             .field("tv_sec", &self.t.t.tv_sec)
             .field("tv_nsec", &self.t.t.tv_nsec)
             .finish()
        }
    }

    #[cfg(not(any(target_os = "dragonfly", target_os = "hermit")))]
    pub type clock_t = libc::c_int;
    #[cfg(any(target_os = "dragonfly", target_os = "hermit"))]
    pub type clock_t = libc::c_ulong;

    fn now(clock: clock_t) -> Timespec {
        let mut t = Timespec {
            t: libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            }
        };
        cvt(unsafe {
            libc::clock_gettime(clock, &mut t.t)
        }).unwrap();
        t
    }
}
