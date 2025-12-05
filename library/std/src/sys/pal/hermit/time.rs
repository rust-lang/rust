#![allow(dead_code)]

use core::hash::{Hash, Hasher};

use super::hermit_abi::{self, CLOCK_MONOTONIC, CLOCK_REALTIME, timespec};
use crate::cmp::Ordering;
use crate::ops::{Add, AddAssign, Sub, SubAssign};
use crate::time::Duration;

const NSEC_PER_SEC: i32 = 1_000_000_000;

#[derive(Copy, Clone, Debug)]
struct Timespec {
    t: timespec,
}

impl Timespec {
    const fn zero() -> Timespec {
        Timespec { t: timespec { tv_sec: 0, tv_nsec: 0 } }
    }

    const fn new(tv_sec: i64, tv_nsec: i32) -> Timespec {
        assert!(tv_nsec >= 0 && tv_nsec < NSEC_PER_SEC);
        // SAFETY: The assert above checks tv_nsec is within the valid range
        Timespec { t: timespec { tv_sec, tv_nsec } }
    }

    fn sub_timespec(&self, other: &Timespec) -> Result<Duration, Duration> {
        fn sub_ge_to_unsigned(a: i64, b: i64) -> u64 {
            debug_assert!(a >= b);
            a.wrapping_sub(b).cast_unsigned()
        }

        if self >= other {
            // Logic here is identical to Unix version of `Timestamp::sub_timespec`,
            // check comments there why operations do not overflow.
            Ok(if self.t.tv_nsec >= other.t.tv_nsec {
                Duration::new(
                    sub_ge_to_unsigned(self.t.tv_sec, other.t.tv_sec),
                    (self.t.tv_nsec - other.t.tv_nsec) as u32,
                )
            } else {
                Duration::new(
                    sub_ge_to_unsigned(self.t.tv_sec - 1, other.t.tv_sec),
                    (self.t.tv_nsec + NSEC_PER_SEC - other.t.tv_nsec) as u32,
                )
            })
        } else {
            match other.sub_timespec(self) {
                Ok(d) => Err(d),
                Err(d) => Ok(d),
            }
        }
    }

    fn checked_add_duration(&self, other: &Duration) -> Option<Timespec> {
        let mut secs = self.t.tv_sec.checked_add_unsigned(other.as_secs())?;

        // Nano calculations can't overflow because nanos are <1B which fit
        // in a u32.
        let mut nsec = other.subsec_nanos() + u32::try_from(self.t.tv_nsec).unwrap();
        if nsec >= NSEC_PER_SEC.try_into().unwrap() {
            nsec -= u32::try_from(NSEC_PER_SEC).unwrap();
            secs = secs.checked_add(1)?;
        }
        Some(Timespec { t: timespec { tv_sec: secs, tv_nsec: nsec as _ } })
    }

    fn checked_sub_duration(&self, other: &Duration) -> Option<Timespec> {
        let mut secs = self.t.tv_sec.checked_sub_unsigned(other.as_secs())?;

        // Similar to above, nanos can't overflow.
        let mut nsec = self.t.tv_nsec as i32 - other.subsec_nanos() as i32;
        if nsec < 0 {
            nsec += NSEC_PER_SEC as i32;
            secs = secs.checked_sub(1)?;
        }
        Some(Timespec { t: timespec { tv_sec: secs, tv_nsec: nsec as _ } })
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
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.t.tv_sec.hash(state);
        self.t.tv_nsec.hash(state);
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(Timespec);

impl Instant {
    pub fn now() -> Instant {
        let mut time: Timespec = Timespec::zero();
        let _ = unsafe { hermit_abi::clock_gettime(CLOCK_MONOTONIC, &raw mut time.t) };

        Instant(time)
    }

    #[stable(feature = "time2", since = "1.8.0")]
    pub fn elapsed(&self) -> Duration {
        Instant::now() - *self
    }

    pub fn duration_since(&self, earlier: Instant) -> Duration {
        self.checked_duration_since(earlier).unwrap_or_default()
    }

    pub fn checked_duration_since(&self, earlier: Instant) -> Option<Duration> {
        self.checked_sub_instant(&earlier)
    }

    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        self.0.sub_timespec(&other.0).ok()
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant(self.0.checked_add_duration(other)?))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant(self.0.checked_sub_duration(other)?))
    }

    pub fn checked_add(&self, duration: Duration) -> Option<Instant> {
        self.0.checked_add_duration(&duration).map(Instant)
    }

    pub fn checked_sub(&self, duration: Duration) -> Option<Instant> {
        self.0.checked_sub_duration(&duration).map(Instant)
    }
}

impl Add<Duration> for Instant {
    type Output = Instant;

    /// # Panics
    ///
    /// This function may panic if the resulting point in time cannot be represented by the
    /// underlying data structure. See [`Instant::checked_add`] for a version without panic.
    fn add(self, other: Duration) -> Instant {
        self.checked_add(other).expect("overflow when adding duration to instant")
    }
}

impl AddAssign<Duration> for Instant {
    fn add_assign(&mut self, other: Duration) {
        *self = *self + other;
    }
}

impl Sub<Duration> for Instant {
    type Output = Instant;

    fn sub(self, other: Duration) -> Instant {
        self.checked_sub(other).expect("overflow when subtracting duration from instant")
    }
}

impl SubAssign<Duration> for Instant {
    fn sub_assign(&mut self, other: Duration) {
        *self = *self - other;
    }
}

impl Sub<Instant> for Instant {
    type Output = Duration;

    /// Returns the amount of time elapsed from another instant to this one,
    /// or zero duration if that instant is later than this one.
    ///
    /// # Panics
    ///
    /// Previous Rust versions panicked when `other` was later than `self`. Currently this
    /// method saturates. Future versions may reintroduce the panic in some circumstances.
    /// See [Monotonicity].
    ///
    /// [Monotonicity]: Instant#monotonicity
    fn sub(self, other: Instant) -> Duration {
        self.duration_since(other)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SystemTime(Timespec);

pub const UNIX_EPOCH: SystemTime = SystemTime(Timespec::zero());

impl SystemTime {
    pub fn new(tv_sec: i64, tv_nsec: i32) -> SystemTime {
        SystemTime(Timespec::new(tv_sec, tv_nsec))
    }

    pub fn now() -> SystemTime {
        let mut time: Timespec = Timespec::zero();
        let _ = unsafe { hermit_abi::clock_gettime(CLOCK_REALTIME, &raw mut time.t) };

        SystemTime(time)
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
