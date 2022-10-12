use crate::fmt;
use crate::time::Duration;

pub use self::inner::Instant;

const NSEC_PER_SEC: u64 = 1_000_000_000;
pub const UNIX_EPOCH: SystemTime = SystemTime { t: Timespec::zero() };
#[allow(dead_code)] // Used for pthread condvar timeouts
pub const TIMESPEC_MAX: libc::timespec =
    libc::timespec { tv_sec: <libc::time_t>::MAX, tv_nsec: 1_000_000_000 - 1 };

// This additional constant is only used when calling
// `libc::pthread_cond_timedwait`.
#[cfg(target_os = "nto")]
pub(super) const TIMESPEC_MAX_CAPPED: libc::timespec = libc::timespec {
    tv_sec: (u64::MAX / NSEC_PER_SEC) as i64,
    tv_nsec: (u64::MAX % NSEC_PER_SEC) as i64,
};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(0)]
#[rustc_layout_scalar_valid_range_end(999_999_999)]
struct Nanoseconds(u32);

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SystemTime {
    pub(in crate::sys::unix) t: Timespec,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::sys::unix) struct Timespec {
    tv_sec: i64,
    tv_nsec: Nanoseconds,
}

impl SystemTime {
    #[cfg_attr(target_os = "horizon", allow(unused))]
    pub fn new(tv_sec: i64, tv_nsec: i64) -> SystemTime {
        SystemTime { t: Timespec::new(tv_sec, tv_nsec) }
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

impl From<libc::timespec> for SystemTime {
    fn from(t: libc::timespec) -> SystemTime {
        SystemTime { t: Timespec::from(t) }
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

impl Timespec {
    pub const fn zero() -> Timespec {
        Timespec::new(0, 0)
    }

    const fn new(tv_sec: i64, tv_nsec: i64) -> Timespec {
        assert!(tv_nsec >= 0 && tv_nsec < NSEC_PER_SEC as i64);
        // SAFETY: The assert above checks tv_nsec is within the valid range
        Timespec { tv_sec, tv_nsec: unsafe { Nanoseconds(tv_nsec as u32) } }
    }

    pub fn sub_timespec(&self, other: &Timespec) -> Result<Duration, Duration> {
        if self >= other {
            // NOTE(eddyb) two aspects of this `if`-`else` are required for LLVM
            // to optimize it into a branchless form (see also #75545):
            //
            // 1. `self.tv_sec - other.tv_sec` shows up as a common expression
            //    in both branches, i.e. the `else` must have its `- 1`
            //    subtraction after the common one, not interleaved with it
            //    (it used to be `self.tv_sec - 1 - other.tv_sec`)
            //
            // 2. the `Duration::new` call (or any other additional complexity)
            //    is outside of the `if`-`else`, not duplicated in both branches
            //
            // Ideally this code could be rearranged such that it more
            // directly expresses the lower-cost behavior we want from it.
            let (secs, nsec) = if self.tv_nsec.0 >= other.tv_nsec.0 {
                ((self.tv_sec - other.tv_sec) as u64, self.tv_nsec.0 - other.tv_nsec.0)
            } else {
                (
                    (self.tv_sec - other.tv_sec - 1) as u64,
                    self.tv_nsec.0 + (NSEC_PER_SEC as u32) - other.tv_nsec.0,
                )
            };

            Ok(Duration::new(secs, nsec))
        } else {
            match other.sub_timespec(self) {
                Ok(d) => Err(d),
                Err(d) => Ok(d),
            }
        }
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Timespec> {
        let mut secs = self.tv_sec.checked_add_unsigned(other.as_secs())?;

        // Nano calculations can't overflow because nanos are <1B which fit
        // in a u32.
        let mut nsec = other.subsec_nanos() + self.tv_nsec.0;
        if nsec >= NSEC_PER_SEC as u32 {
            nsec -= NSEC_PER_SEC as u32;
            secs = secs.checked_add(1)?;
        }
        Some(Timespec::new(secs, nsec.into()))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Timespec> {
        let mut secs = self.tv_sec.checked_sub_unsigned(other.as_secs())?;

        // Similar to above, nanos can't overflow.
        let mut nsec = self.tv_nsec.0 as i32 - other.subsec_nanos() as i32;
        if nsec < 0 {
            nsec += NSEC_PER_SEC as i32;
            secs = secs.checked_sub(1)?;
        }
        Some(Timespec::new(secs, nsec.into()))
    }

    #[allow(dead_code)]
    pub fn to_timespec(&self) -> Option<libc::timespec> {
        Some(libc::timespec {
            tv_sec: self.tv_sec.try_into().ok()?,
            tv_nsec: self.tv_nsec.0.try_into().ok()?,
        })
    }

    // On QNX Neutrino, the maximum timespec for e.g. pthread_cond_timedwait
    // is 2^64 nanoseconds
    #[cfg(target_os = "nto")]
    pub(super) fn to_timespec_capped(&self) -> Option<libc::timespec> {
        // Check if timeout in nanoseconds would fit into an u64
        if (self.tv_nsec.0 as u64)
            .checked_add((self.tv_sec as u64).checked_mul(NSEC_PER_SEC)?)
            .is_none()
        {
            return None;
        }
        self.to_timespec()
    }

    #[cfg(all(
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "32",
        not(target_arch = "riscv32")
    ))]
    pub fn to_timespec64(&self) -> __timespec64 {
        __timespec64::new(self.tv_sec, self.tv_nsec.0 as _)
    }
}

impl From<libc::timespec> for Timespec {
    fn from(t: libc::timespec) -> Timespec {
        Timespec::new(t.tv_sec as i64, t.tv_nsec as i64)
    }
}

#[cfg(all(
    target_os = "linux",
    target_env = "gnu",
    target_pointer_width = "32",
    not(target_arch = "riscv32")
))]
#[repr(C)]
pub(in crate::sys::unix) struct __timespec64 {
    pub(in crate::sys::unix) tv_sec: i64,
    #[cfg(target_endian = "big")]
    _padding: i32,
    pub(in crate::sys::unix) tv_nsec: i32,
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
    pub(in crate::sys::unix) fn new(tv_sec: i64, tv_nsec: i32) -> Self {
        Self { tv_sec, tv_nsec, _padding: 0 }
    }
}

#[cfg(all(
    target_os = "linux",
    target_env = "gnu",
    target_pointer_width = "32",
    not(target_arch = "riscv32")
))]
impl From<__timespec64> for Timespec {
    fn from(t: __timespec64) -> Timespec {
        Timespec::new(t.tv_sec, t.tv_nsec.into())
    }
}

#[cfg(any(
    all(target_os = "macos", any(not(target_arch = "aarch64"))),
    target_os = "ios",
    target_os = "watchos",
    target_os = "tvos"
))]
mod inner {
    use crate::sync::atomic::{AtomicU64, Ordering};
    use crate::sys::cvt;
    use crate::sys_common::mul_div_u64;
    use crate::time::Duration;

    use super::{SystemTime, Timespec, NSEC_PER_SEC};

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
    pub struct Instant {
        t: u64,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct mach_timebase_info {
        numer: u32,
        denom: u32,
    }
    type mach_timebase_info_t = *mut mach_timebase_info;
    type kern_return_t = libc::c_int;

    impl Instant {
        pub fn now() -> Instant {
            extern "C" {
                fn mach_absolute_time() -> u64;
            }
            Instant { t: unsafe { mach_absolute_time() } }
        }

        pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
            let diff = self.t.checked_sub(other.t)?;
            let info = info();
            let nanos = mul_div_u64(diff, info.numer as u64, info.denom as u64);
            Some(Duration::new(nanos / NSEC_PER_SEC, (nanos % NSEC_PER_SEC) as u32))
        }

        pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
            Some(Instant { t: self.t.checked_add(checked_dur2intervals(other)?)? })
        }

        pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
            Some(Instant { t: self.t.checked_sub(checked_dur2intervals(other)?)? })
        }
    }

    impl SystemTime {
        pub fn now() -> SystemTime {
            use crate::ptr;

            let mut s = libc::timeval { tv_sec: 0, tv_usec: 0 };
            cvt(unsafe { libc::gettimeofday(&mut s, ptr::null_mut()) }).unwrap();
            return SystemTime::from(s);
        }
    }

    impl From<libc::timeval> for Timespec {
        fn from(t: libc::timeval) -> Timespec {
            Timespec::new(t.tv_sec as i64, 1000 * t.tv_usec as i64)
        }
    }

    impl From<libc::timeval> for SystemTime {
        fn from(t: libc::timeval) -> SystemTime {
            SystemTime { t: Timespec::from(t) }
        }
    }

    fn checked_dur2intervals(dur: &Duration) -> Option<u64> {
        let nanos =
            dur.as_secs().checked_mul(NSEC_PER_SEC)?.checked_add(dur.subsec_nanos() as u64)?;
        let info = info();
        Some(mul_div_u64(nanos, info.denom as u64, info.numer as u64))
    }

    fn info() -> mach_timebase_info {
        // INFO_BITS conceptually is an `Option<mach_timebase_info>`. We can do
        // this in 64 bits because we know 0 is never a valid value for the
        // `denom` field.
        //
        // Encoding this as a single `AtomicU64` allows us to use `Relaxed`
        // operations, as we are only interested in the effects on a single
        // memory location.
        static INFO_BITS: AtomicU64 = AtomicU64::new(0);

        // If a previous thread has initialized `INFO_BITS`, use it.
        let info_bits = INFO_BITS.load(Ordering::Relaxed);
        if info_bits != 0 {
            return info_from_bits(info_bits);
        }

        // ... otherwise learn for ourselves ...
        extern "C" {
            fn mach_timebase_info(info: mach_timebase_info_t) -> kern_return_t;
        }

        let mut info = info_from_bits(0);
        unsafe {
            mach_timebase_info(&mut info);
        }
        INFO_BITS.store(info_to_bits(info), Ordering::Relaxed);
        info
    }

    #[inline]
    fn info_to_bits(info: mach_timebase_info) -> u64 {
        ((info.denom as u64) << 32) | (info.numer as u64)
    }

    #[inline]
    fn info_from_bits(bits: u64) -> mach_timebase_info {
        mach_timebase_info { numer: bits as u32, denom: (bits >> 32) as u32 }
    }
}

#[cfg(not(any(
    all(target_os = "macos", any(not(target_arch = "aarch64"))),
    target_os = "ios",
    target_os = "watchos",
    target_os = "tvos"
)))]
mod inner {
    use crate::fmt;
    use crate::mem::MaybeUninit;
    use crate::sys::cvt;
    use crate::time::Duration;

    use super::{SystemTime, Timespec};

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Instant {
        t: Timespec,
    }

    impl Instant {
        pub fn now() -> Instant {
            #[cfg(target_os = "macos")]
            const clock_id: libc::clockid_t = libc::CLOCK_UPTIME_RAW;
            #[cfg(not(target_os = "macos"))]
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

    impl SystemTime {
        pub fn now() -> SystemTime {
            SystemTime { t: Timespec::now(libc::CLOCK_REALTIME) }
        }
    }

    impl Timespec {
        pub fn now(clock: libc::clockid_t) -> Timespec {
            // Try to use 64-bit time in preparation for Y2038.
            #[cfg(all(
                target_os = "linux",
                target_env = "gnu",
                target_pointer_width = "32",
                not(target_arch = "riscv32")
            ))]
            {
                use crate::sys::weak::weak;

                // __clock_gettime64 was added to 32-bit arches in glibc 2.34,
                // and it handles both vDSO calls and ENOSYS fallbacks itself.
                weak!(fn __clock_gettime64(libc::clockid_t, *mut super::__timespec64) -> libc::c_int);

                if let Some(clock_gettime64) = __clock_gettime64.get() {
                    let mut t = MaybeUninit::uninit();
                    cvt(unsafe { clock_gettime64(clock, t.as_mut_ptr()) }).unwrap();
                    return Timespec::from(unsafe { t.assume_init() });
                }
            }

            let mut t = MaybeUninit::uninit();
            cvt(unsafe { libc::clock_gettime(clock, t.as_mut_ptr()) }).unwrap();
            Timespec::from(unsafe { t.assume_init() })
        }
    }
}
