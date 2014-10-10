// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Temporal quantification

#![experimental]

use {fmt, i64};
use ops::{Add, Sub, Mul, Div, Neg};
use option::{Option, Some, None};
use num;
use num::{CheckedAdd, CheckedMul};
use result::{Result, Ok, Err};

/// The number of nanoseconds in a microsecond.
const NANOS_PER_MICRO: i32 = 1000;
/// The number of nanoseconds in a millisecond.
const NANOS_PER_MILLI: i32 = 1000_000;
/// The number of nanoseconds in seconds.
const NANOS_PER_SEC: i32 = 1_000_000_000;
/// The number of microseconds per second.
const MICROS_PER_SEC: i64 = 1000_000;
/// The number of milliseconds per second.
const MILLIS_PER_SEC: i64 = 1000;
/// The number of seconds in a minute.
const SECS_PER_MINUTE: i64 = 60;
/// The number of seconds in an hour.
const SECS_PER_HOUR: i64 = 3600;
/// The number of (non-leap) seconds in days.
const SECS_PER_DAY: i64 = 86400;
/// The number of (non-leap) seconds in a week.
const SECS_PER_WEEK: i64 = 604800;

macro_rules! try_opt(
    ($e:expr) => (match $e { Some(v) => v, None => return None })
)


/// ISO 8601 time duration with nanosecond precision.
/// This also allows for the negative duration; see individual methods for details.
#[deriving(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Duration {
    secs: i64,
    nanos: i32, // Always 0 <= nanos < NANOS_PER_SEC
}

/// The minimum possible `Duration`: `i64::MIN` milliseconds.
pub const MIN: Duration = Duration {
    secs: i64::MIN / MILLIS_PER_SEC - 1,
    nanos: NANOS_PER_SEC + (i64::MIN % MILLIS_PER_SEC) as i32 * NANOS_PER_MILLI
};

/// The maximum possible `Duration`: `i64::MAX` milliseconds.
pub const MAX: Duration = Duration {
    secs: i64::MAX / MILLIS_PER_SEC,
    nanos: (i64::MAX % MILLIS_PER_SEC) as i32 * NANOS_PER_MILLI
};

impl Duration {
    /// Makes a new `Duration` with given number of weeks.
    /// Equivalent to `Duration::seconds(weeks * 7 * 24 * 60 * 60), with overflow checks.
    /// Fails when the duration is out of bounds.
    #[inline]
    pub fn weeks(weeks: i64) -> Duration {
        let secs = weeks.checked_mul(&SECS_PER_WEEK).expect("Duration::weeks out of bounds");
        Duration::seconds(secs)
    }

    /// Makes a new `Duration` with given number of days.
    /// Equivalent to `Duration::seconds(days * 24 * 60 * 60)` with overflow checks.
    /// Fails when the duration is out of bounds.
    #[inline]
    pub fn days(days: i64) -> Duration {
        let secs = days.checked_mul(&SECS_PER_DAY).expect("Duration::days out of bounds");
        Duration::seconds(secs)
    }

    /// Makes a new `Duration` with given number of hours.
    /// Equivalent to `Duration::seconds(hours * 60 * 60)` with overflow checks.
    /// Fails when the duration is out of bounds.
    #[inline]
    pub fn hours(hours: i64) -> Duration {
        let secs = hours.checked_mul(&SECS_PER_HOUR).expect("Duration::hours ouf of bounds");
        Duration::seconds(secs)
    }

    /// Makes a new `Duration` with given number of minutes.
    /// Equivalent to `Duration::seconds(minutes * 60)` with overflow checks.
    /// Fails when the duration is out of bounds.
    #[inline]
    pub fn minutes(minutes: i64) -> Duration {
        let secs = minutes.checked_mul(&SECS_PER_MINUTE).expect("Duration::minutes out of bounds");
        Duration::seconds(secs)
    }

    /// Makes a new `Duration` with given number of seconds.
    /// Fails when the duration is more than `i64::MAX` milliseconds
    /// or less than `i64::MIN` milliseconds.
    #[inline]
    pub fn seconds(seconds: i64) -> Duration {
        let d = Duration { secs: seconds, nanos: 0 };
        if d < MIN || d > MAX {
            fail!("Duration::seconds out of bounds");
        }
        d
    }

    /// Makes a new `Duration` with given number of milliseconds.
    #[inline]
    pub fn milliseconds(milliseconds: i64) -> Duration {
        let (secs, millis) = div_mod_floor_64(milliseconds, MILLIS_PER_SEC);
        let nanos = millis as i32 * NANOS_PER_MILLI;
        Duration { secs: secs, nanos: nanos }
    }

    /// Makes a new `Duration` with given number of microseconds.
    #[inline]
    pub fn microseconds(microseconds: i64) -> Duration {
        let (secs, micros) = div_mod_floor_64(microseconds, MICROS_PER_SEC);
        let nanos = micros as i32 * NANOS_PER_MICRO;
        Duration { secs: secs, nanos: nanos }
    }

    /// Makes a new `Duration` with given number of nanoseconds.
    #[inline]
    pub fn nanoseconds(nanos: i64) -> Duration {
        let (secs, nanos) = div_mod_floor_64(nanos, NANOS_PER_SEC as i64);
        Duration { secs: secs, nanos: nanos as i32 }
    }

    /// Returns the total number of whole weeks in the duration.
    #[inline]
    pub fn num_weeks(&self) -> i64 {
        self.num_days() / 7
    }

    /// Returns the total number of whole days in the duration.
    pub fn num_days(&self) -> i64 {
        self.num_seconds() / SECS_PER_DAY
    }

    /// Returns the total number of whole hours in the duration.
    #[inline]
    pub fn num_hours(&self) -> i64 {
        self.num_seconds() / SECS_PER_HOUR
    }

    /// Returns the total number of whole minutes in the duration.
    #[inline]
    pub fn num_minutes(&self) -> i64 {
        self.num_seconds() / SECS_PER_MINUTE
    }

    /// Returns the total number of whole seconds in the duration.
    pub fn num_seconds(&self) -> i64 {
        // If secs is negative, nanos should be subtracted from the duration.
        if self.secs < 0 && self.nanos > 0 {
            self.secs + 1
        } else {
            self.secs
        }
    }

    /// Returns the number of nanoseconds such that
    /// `nanos_mod_sec() + num_seconds() * NANOS_PER_SEC` is the total number of
    /// nanoseconds in the duration.
    fn nanos_mod_sec(&self) -> i32 {
        if self.secs < 0 && self.nanos > 0 {
            self.nanos - NANOS_PER_SEC
        } else {
            self.nanos
        }
    }

    /// Returns the total number of whole milliseconds in the duration,
    pub fn num_milliseconds(&self) -> i64 {
        // A proper Duration will not overflow, because MIN and MAX are defined
        // such that the range is exactly i64 milliseconds.
        let secs_part = self.num_seconds() * MILLIS_PER_SEC;
        let nanos_part = self.nanos_mod_sec() / NANOS_PER_MILLI;
        secs_part + nanos_part as i64
    }

    /// Returns the total number of whole microseconds in the duration,
    /// or `None` on overflow (exceeding 2^63 microseconds in either direction).
    pub fn num_microseconds(&self) -> Option<i64> {
        let secs_part = try_opt!(self.num_seconds().checked_mul(&MICROS_PER_SEC));
        let nanos_part = self.nanos_mod_sec() / NANOS_PER_MICRO;
        secs_part.checked_add(&(nanos_part as i64))
    }

    /// Returns the total number of whole nanoseconds in the duration,
    /// or `None` on overflow (exceeding 2^63 nanoseconds in either direction).
    pub fn num_nanoseconds(&self) -> Option<i64> {
        let secs_part = try_opt!(self.num_seconds().checked_mul(&(NANOS_PER_SEC as i64)));
        let nanos_part = self.nanos_mod_sec();
        secs_part.checked_add(&(nanos_part as i64))
    }
}

impl num::Bounded for Duration {
    #[inline] fn min_value() -> Duration { MIN }
    #[inline] fn max_value() -> Duration { MAX }
}

impl num::Zero for Duration {
    #[inline]
    fn zero() -> Duration {
        Duration { secs: 0, nanos: 0 }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.secs == 0 && self.nanos == 0
    }
}

impl Neg<Duration> for Duration {
    #[inline]
    fn neg(&self) -> Duration {
        if self.nanos == 0 {
            Duration { secs: -self.secs, nanos: 0 }
        } else {
            Duration { secs: -self.secs - 1, nanos: NANOS_PER_SEC - self.nanos }
        }
    }
}

impl Add<Duration,Duration> for Duration {
    fn add(&self, rhs: &Duration) -> Duration {
        let mut secs = self.secs + rhs.secs;
        let mut nanos = self.nanos + rhs.nanos;
        if nanos >= NANOS_PER_SEC {
            nanos -= NANOS_PER_SEC;
            secs += 1;
        }
        Duration { secs: secs, nanos: nanos }
    }
}

impl num::CheckedAdd for Duration {
    fn checked_add(&self, rhs: &Duration) -> Option<Duration> {
        let mut secs = try_opt!(self.secs.checked_add(&rhs.secs));
        let mut nanos = self.nanos + rhs.nanos;
        if nanos >= NANOS_PER_SEC {
            nanos -= NANOS_PER_SEC;
            secs = try_opt!(secs.checked_add(&1));
        }
        let d = Duration { secs: secs, nanos: nanos };
        // Even if d is within the bounds of i64 seconds,
        // it might still overflow i64 milliseconds.
        if d < MIN || d > MAX { None } else { Some(d) }
    }
}

impl Sub<Duration,Duration> for Duration {
    fn sub(&self, rhs: &Duration) -> Duration {
        let mut secs = self.secs - rhs.secs;
        let mut nanos = self.nanos - rhs.nanos;
        if nanos < 0 {
            nanos += NANOS_PER_SEC;
            secs -= 1;
        }
        Duration { secs: secs, nanos: nanos }
    }
}

impl num::CheckedSub for Duration {
    fn checked_sub(&self, rhs: &Duration) -> Option<Duration> {
        let mut secs = try_opt!(self.secs.checked_sub(&rhs.secs));
        let mut nanos = self.nanos - rhs.nanos;
        if nanos < 0 {
            nanos += NANOS_PER_SEC;
            secs = try_opt!(secs.checked_sub(&1));
        }
        let d = Duration { secs: secs, nanos: nanos };
        // Even if d is within the bounds of i64 seconds,
        // it might still overflow i64 milliseconds.
        if d < MIN || d > MAX { None } else { Some(d) }
    }
}

impl Mul<i32,Duration> for Duration {
    fn mul(&self, rhs: &i32) -> Duration {
        // Multiply nanoseconds as i64, because it cannot overflow that way.
        let total_nanos = self.nanos as i64 * *rhs as i64;
        let (extra_secs, nanos) = div_mod_floor_64(total_nanos, NANOS_PER_SEC as i64);
        let secs = self.secs * *rhs as i64 + extra_secs;
        Duration { secs: secs, nanos: nanos as i32 }
    }
}

impl Div<i32,Duration> for Duration {
    fn div(&self, rhs: &i32) -> Duration {
        let mut secs = self.secs / *rhs as i64;
        let carry = self.secs - secs * *rhs as i64;
        let extra_nanos = carry * NANOS_PER_SEC as i64 / *rhs as i64;
        let mut nanos = self.nanos / *rhs + extra_nanos as i32;
        if nanos >= NANOS_PER_SEC {
            nanos -= NANOS_PER_SEC;
            secs += 1;
        }
        if nanos < 0 {
            nanos += NANOS_PER_SEC;
            secs -= 1;
        }
        Duration { secs: secs, nanos: nanos }
    }
}

impl fmt::Show for Duration {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let days = self.num_days();
        let secs = self.secs - days * SECS_PER_DAY;
        let hasdate = days != 0;
        let hastime = (secs != 0 || self.nanos != 0) || !hasdate;

        try!(write!(f, "P"));
        if hasdate {
            // technically speaking the negative part is not the valid ISO 8601,
            // but we need to print it anyway.
            try!(write!(f, "{}D", days));
        }
        if hastime {
            if self.nanos == 0 {
                try!(write!(f, "T{}S", secs));
            } else if self.nanos % NANOS_PER_MILLI == 0 {
                try!(write!(f, "T{}.{:03}S", secs, self.nanos / NANOS_PER_MILLI));
            } else if self.nanos % NANOS_PER_MICRO == 0 {
                try!(write!(f, "T{}.{:06}S", secs, self.nanos / NANOS_PER_MICRO));
            } else {
                try!(write!(f, "T{}.{:09}S", secs, self.nanos));
            }
        }
        Ok(())
    }
}

// Copied from libnum
#[inline]
fn div_mod_floor_64(this: i64, other: i64) -> (i64, i64) {
    (div_floor_64(this, other), mod_floor_64(this, other))
}

#[inline]
fn div_floor_64(this: i64, other: i64) -> i64 {
    match div_rem_64(this, other) {
        (d, r) if (r > 0 && other < 0)
               || (r < 0 && other > 0) => d - 1,
        (d, _)                         => d,
    }
}

#[inline]
fn mod_floor_64(this: i64, other: i64) -> i64 {
    match this % other {
        r if (r > 0 && other < 0)
          || (r < 0 && other > 0) => r + other,
        r                         => r,
    }
}

#[inline]
fn div_rem_64(this: i64, other: i64) -> (i64, i64) {
    (this / other, this % other)
}

#[cfg(test)]
mod tests {
    use super::{Duration, MIN, MAX};
    use {i32, i64};
    use num::{Zero, CheckedAdd, CheckedSub};
    use option::{Some, None};
    use to_string::ToString;

    #[test]
    fn test_duration() {
        let d: Duration = Zero::zero();
        assert_eq!(d, Zero::zero());
        assert!(Duration::seconds(1) != Zero::zero());
        assert_eq!(Duration::seconds(1) + Duration::seconds(2), Duration::seconds(3));
        assert_eq!(Duration::seconds(86399) + Duration::seconds(4),
                   Duration::days(1) + Duration::seconds(3));
        assert_eq!(Duration::days(10) - Duration::seconds(1000), Duration::seconds(863000));
        assert_eq!(Duration::days(10) - Duration::seconds(1000000), Duration::seconds(-136000));
        assert_eq!(Duration::days(2) + Duration::seconds(86399) +
                   Duration::nanoseconds(1234567890),
                   Duration::days(3) + Duration::nanoseconds(234567890));
        assert_eq!(-Duration::days(3), Duration::days(-3));
        assert_eq!(-(Duration::days(3) + Duration::seconds(70)),
                   Duration::days(-4) + Duration::seconds(86400-70));
    }

    #[test]
    fn test_duration_num_days() {
        let d: Duration = Zero::zero();
        assert_eq!(d.num_days(), 0);
        assert_eq!(Duration::days(1).num_days(), 1);
        assert_eq!(Duration::days(-1).num_days(), -1);
        assert_eq!(Duration::seconds(86399).num_days(), 0);
        assert_eq!(Duration::seconds(86401).num_days(), 1);
        assert_eq!(Duration::seconds(-86399).num_days(), 0);
        assert_eq!(Duration::seconds(-86401).num_days(), -1);
        assert_eq!(Duration::days(i32::MAX as i64).num_days(), i32::MAX as i64);
        assert_eq!(Duration::days(i32::MIN as i64).num_days(), i32::MIN as i64);
    }

    #[test]
    fn test_duration_num_seconds() {
        let d: Duration = Zero::zero();
        assert_eq!(d.num_seconds(), 0);
        assert_eq!(Duration::seconds(1).num_seconds(), 1);
        assert_eq!(Duration::seconds(-1).num_seconds(), -1);
        assert_eq!(Duration::milliseconds(999).num_seconds(), 0);
        assert_eq!(Duration::milliseconds(1001).num_seconds(), 1);
        assert_eq!(Duration::milliseconds(-999).num_seconds(), 0);
        assert_eq!(Duration::milliseconds(-1001).num_seconds(), -1);
    }

    #[test]
    fn test_duration_num_milliseconds() {
        let d: Duration = Zero::zero();
        assert_eq!(d.num_milliseconds(), 0);
        assert_eq!(Duration::milliseconds(1).num_milliseconds(), 1);
        assert_eq!(Duration::milliseconds(-1).num_milliseconds(), -1);
        assert_eq!(Duration::microseconds(999).num_milliseconds(), 0);
        assert_eq!(Duration::microseconds(1001).num_milliseconds(), 1);
        assert_eq!(Duration::microseconds(-999).num_milliseconds(), 0);
        assert_eq!(Duration::microseconds(-1001).num_milliseconds(), -1);
        assert_eq!(Duration::milliseconds(i64::MAX).num_milliseconds(), i64::MAX);
        assert_eq!(Duration::milliseconds(i64::MIN).num_milliseconds(), i64::MIN);
        assert_eq!(MAX.num_milliseconds(), i64::MAX);
        assert_eq!(MIN.num_milliseconds(), i64::MIN);
    }

    #[test]
    fn test_duration_num_microseconds() {
        let d: Duration = Zero::zero();
        assert_eq!(d.num_microseconds(), Some(0));
        assert_eq!(Duration::microseconds(1).num_microseconds(), Some(1));
        assert_eq!(Duration::microseconds(-1).num_microseconds(), Some(-1));
        assert_eq!(Duration::nanoseconds(999).num_microseconds(), Some(0));
        assert_eq!(Duration::nanoseconds(1001).num_microseconds(), Some(1));
        assert_eq!(Duration::nanoseconds(-999).num_microseconds(), Some(0));
        assert_eq!(Duration::nanoseconds(-1001).num_microseconds(), Some(-1));
        assert_eq!(Duration::microseconds(i64::MAX).num_microseconds(), Some(i64::MAX));
        assert_eq!(Duration::microseconds(i64::MIN).num_microseconds(), Some(i64::MIN));
        assert_eq!(MAX.num_microseconds(), None);
        assert_eq!(MIN.num_microseconds(), None);

        // overflow checks
        const MICROS_PER_DAY: i64 = 86400_000_000;
        assert_eq!(Duration::days(i64::MAX / MICROS_PER_DAY).num_microseconds(),
                   Some(i64::MAX / MICROS_PER_DAY * MICROS_PER_DAY));
        assert_eq!(Duration::days(i64::MIN / MICROS_PER_DAY).num_microseconds(),
                   Some(i64::MIN / MICROS_PER_DAY * MICROS_PER_DAY));
        assert_eq!(Duration::days(i64::MAX / MICROS_PER_DAY + 1).num_microseconds(), None);
        assert_eq!(Duration::days(i64::MIN / MICROS_PER_DAY - 1).num_microseconds(), None);
    }

    #[test]
    fn test_duration_num_nanoseconds() {
        let d: Duration = Zero::zero();
        assert_eq!(d.num_nanoseconds(), Some(0));
        assert_eq!(Duration::nanoseconds(1).num_nanoseconds(), Some(1));
        assert_eq!(Duration::nanoseconds(-1).num_nanoseconds(), Some(-1));
        assert_eq!(Duration::nanoseconds(i64::MAX).num_nanoseconds(), Some(i64::MAX));
        assert_eq!(Duration::nanoseconds(i64::MIN).num_nanoseconds(), Some(i64::MIN));
        assert_eq!(MAX.num_nanoseconds(), None);
        assert_eq!(MIN.num_nanoseconds(), None);

        // overflow checks
        const NANOS_PER_DAY: i64 = 86400_000_000_000;
        assert_eq!(Duration::days(i64::MAX / NANOS_PER_DAY).num_nanoseconds(),
                   Some(i64::MAX / NANOS_PER_DAY * NANOS_PER_DAY));
        assert_eq!(Duration::days(i64::MIN / NANOS_PER_DAY).num_nanoseconds(),
                   Some(i64::MIN / NANOS_PER_DAY * NANOS_PER_DAY));
        assert_eq!(Duration::days(i64::MAX / NANOS_PER_DAY + 1).num_nanoseconds(), None);
        assert_eq!(Duration::days(i64::MIN / NANOS_PER_DAY - 1).num_nanoseconds(), None);
    }

    #[test]
    fn test_duration_checked_ops() {
        assert_eq!(Duration::milliseconds(i64::MAX - 1).checked_add(&Duration::microseconds(999)),
                   Some(Duration::milliseconds(i64::MAX - 2) + Duration::microseconds(1999)));
        assert!(Duration::milliseconds(i64::MAX).checked_add(&Duration::microseconds(1000))
                                                .is_none());

        assert_eq!(Duration::milliseconds(i64::MIN).checked_sub(&Duration::milliseconds(0)),
                   Some(Duration::milliseconds(i64::MIN)));
        assert!(Duration::milliseconds(i64::MIN).checked_sub(&Duration::milliseconds(1))
                                                .is_none());
    }

    #[test]
    fn test_duration_mul() {
        let d: Duration = Zero::zero();
        assert_eq!(d * i32::MAX, d);
        assert_eq!(d * i32::MIN, d);
        assert_eq!(Duration::nanoseconds(1) * 0, Zero::zero());
        assert_eq!(Duration::nanoseconds(1) * 1, Duration::nanoseconds(1));
        assert_eq!(Duration::nanoseconds(1) * 1_000_000_000, Duration::seconds(1));
        assert_eq!(Duration::nanoseconds(1) * -1_000_000_000, -Duration::seconds(1));
        assert_eq!(-Duration::nanoseconds(1) * 1_000_000_000, -Duration::seconds(1));
        assert_eq!(Duration::nanoseconds(30) * 333_333_333,
                   Duration::seconds(10) - Duration::nanoseconds(10));
        assert_eq!((Duration::nanoseconds(1) + Duration::seconds(1) + Duration::days(1)) * 3,
                   Duration::nanoseconds(3) + Duration::seconds(3) + Duration::days(3));
        assert_eq!(Duration::milliseconds(1500) * -2, Duration::seconds(-3));
        assert_eq!(Duration::milliseconds(-1500) * 2, Duration::seconds(-3));
    }

    #[test]
    fn test_duration_div() {
        let d: Duration = Zero::zero();
        assert_eq!(d / i32::MAX, d);
        assert_eq!(d / i32::MIN, d);
        assert_eq!(Duration::nanoseconds(123_456_789) / 1, Duration::nanoseconds(123_456_789));
        assert_eq!(Duration::nanoseconds(123_456_789) / -1, -Duration::nanoseconds(123_456_789));
        assert_eq!(-Duration::nanoseconds(123_456_789) / -1, Duration::nanoseconds(123_456_789));
        assert_eq!(-Duration::nanoseconds(123_456_789) / 1, -Duration::nanoseconds(123_456_789));
        assert_eq!(Duration::seconds(1) / 3, Duration::nanoseconds(333_333_333));
        assert_eq!(Duration::seconds(4) / 3, Duration::nanoseconds(1_333_333_333));
        assert_eq!(Duration::seconds(-1) / 2, Duration::milliseconds(-500));
        assert_eq!(Duration::seconds(1) / -2, Duration::milliseconds(-500));
        assert_eq!(Duration::seconds(-1) / -2, Duration::milliseconds(500));
        assert_eq!(Duration::seconds(-4) / 3, Duration::nanoseconds(-1_333_333_333));
        assert_eq!(Duration::seconds(-4) / -3, Duration::nanoseconds(1_333_333_333));
    }

    #[test]
    fn test_duration_fmt() {
        let d: Duration = Zero::zero();
        assert_eq!(d.to_string(), "PT0S".to_string());
        assert_eq!(Duration::days(42).to_string(), "P42D".to_string());
        assert_eq!(Duration::days(-42).to_string(), "P-42D".to_string());
        assert_eq!(Duration::seconds(42).to_string(), "PT42S".to_string());
        assert_eq!(Duration::milliseconds(42).to_string(), "PT0.042S".to_string());
        assert_eq!(Duration::microseconds(42).to_string(), "PT0.000042S".to_string());
        assert_eq!(Duration::nanoseconds(42).to_string(), "PT0.000000042S".to_string());
        assert_eq!((Duration::days(7) + Duration::milliseconds(6543)).to_string(),
                   "P7DT6.543S".to_string());

        // the format specifier should have no effect on `Duration`
        assert_eq!(format!("{:30}", Duration::days(1) + Duration::milliseconds(2345)),
                   "P1DT2.345S".to_string());
    }
}
