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

use {fmt, i32};
use ops::{Add, Sub, Mul, Div, Neg};
use option::{Option, Some, None};
use num;
use num::{CheckedAdd, CheckedMul};
use result::{Result, Ok, Err};


/// `Duration`'s `days` component should have no more than this value.
static MIN_DAYS: i32 = i32::MIN;
/// `Duration`'s `days` component should have no less than this value.
static MAX_DAYS: i32 = i32::MAX;

/// The number of nanoseconds in seconds.
static NANOS_PER_SEC: i32 = 1_000_000_000;
/// The number of (non-leap) seconds in days.
static SECS_PER_DAY: i32 = 86400;

macro_rules! try_opt(
    ($e:expr) => (match $e { Some(v) => v, None => return None })
)


// FIXME #16466: This could be represented as (i64 seconds, u32 nanos)
/// ISO 8601 time duration with nanosecond precision.
/// This also allows for the negative duration; see individual methods for details.
#[deriving(PartialEq, Eq, PartialOrd, Ord)]
pub struct Duration {
    days: i32,
    secs: u32,  // Always < SECS_PER_DAY
    nanos: u32, // Always < NANOS_PR_SECOND
}

/// The minimum possible `Duration`.
pub static MIN: Duration = Duration { days: MIN_DAYS, secs: 0, nanos: 0 };
/// The maximum possible `Duration`.
pub static MAX: Duration = Duration { days: MAX_DAYS, secs: SECS_PER_DAY as u32 - 1,
                                      nanos: NANOS_PER_SEC as u32 - 1 };

impl Duration {
    /// Makes a new `Duration` with given number of weeks.
    /// Equivalent to `Duration::new(weeks * 7, 0, 0)` with overflow checks.
    ///
    /// Fails when the duration is out of bounds.
    #[inline]
    pub fn weeks(weeks: i32) -> Duration {
        let days = weeks.checked_mul(&7).expect("Duration::weeks out of bounds");
        Duration::days(days)
    }

    /// Makes a new `Duration` with given number of days.
    /// Equivalent to `Duration::new(days, 0, 0)`.
    #[inline]
    pub fn days(days: i32) -> Duration {
        Duration { days: days, secs: 0, nanos: 0 }
    }

    /// Makes a new `Duration` with given number of hours.
    /// Equivalent to `Duration::new(0, hours * 3600, 0)` with overflow checks.
    #[inline]
    pub fn hours(hours: i32) -> Duration {
        let (days, hours) = div_mod_floor(hours, (SECS_PER_DAY / 3600));
        let secs = hours * 3600;
        Duration { secs: secs as u32, ..Duration::days(days) }
    }

    /// Makes a new `Duration` with given number of minutes.
    /// Equivalent to `Duration::new(0, mins * 60, 0)` with overflow checks.
    #[inline]
    pub fn minutes(mins: i32) -> Duration {
        let (days, mins) = div_mod_floor(mins, (SECS_PER_DAY / 60));
        let secs = mins * 60;
        Duration { secs: secs as u32, ..Duration::days(days) }
    }

    /// Makes a new `Duration` with given number of seconds.
    /// Equivalent to `Duration::new(0, secs, 0)`.
    #[inline]
    pub fn seconds(secs: i32) -> Duration {
        let (days, secs) = div_mod_floor(secs, SECS_PER_DAY);
        Duration { secs: secs as u32, ..Duration::days(days) }
    }

    /// Makes a new `Duration` with given number of milliseconds.
    /// Equivalent to `Duration::new(0, 0, millis * 1_000_000)` with overflow checks.
    #[inline]
    pub fn milliseconds(millis: i32) -> Duration {
        let (secs, millis) = div_mod_floor(millis, (NANOS_PER_SEC / 1_000_000));
        let nanos = millis * 1_000_000;
        Duration { nanos: nanos as u32, ..Duration::seconds(secs) }
    }

    /// Makes a new `Duration` with given number of microseconds.
    /// Equivalent to `Duration::new(0, 0, micros * 1_000)` with overflow checks.
    #[inline]
    pub fn microseconds(micros: i32) -> Duration {
        let (secs, micros) = div_mod_floor(micros, (NANOS_PER_SEC / 1_000));
        let nanos = micros * 1_000;
        Duration { nanos: nanos as u32, ..Duration::seconds(secs) }
    }

    /// Makes a new `Duration` with given number of nanoseconds.
    /// Equivalent to `Duration::new(0, 0, nanos)`.
    #[inline]
    pub fn nanoseconds(nanos: i32) -> Duration {
        let (secs, nanos) = div_mod_floor(nanos, NANOS_PER_SEC);
        Duration { nanos: nanos as u32, ..Duration::seconds(secs) }
    }

    /// Returns a tuple of the number of days, (non-leap) seconds and
    /// nanoseconds in the duration.  Note that the number of seconds
    /// and nanoseconds are always positive, so that for example
    /// `-Duration::seconds(3)` has -1 days and 86,397 seconds.
    #[inline]
    fn to_tuple_64(&self) -> (i64, u32, u32) {
        (self.days as i64, self.secs, self.nanos)
    }

    /// Negates the duration and returns a tuple like `to_tuple`.
    /// This does not overflow and thus is internally used for several methods.
    fn to_negated_tuple_64(&self) -> (i64, u32, u32) {
        let mut days = -(self.days as i64);
        let mut secs = -(self.secs as i32);
        let mut nanos = -(self.nanos as i32);
        if nanos < 0 {
            nanos += NANOS_PER_SEC;
            secs -= 1;
        }
        if secs < 0 {
            secs += SECS_PER_DAY;
            days -= 1;
        }
        (days, secs as u32, nanos as u32)
    }

    /// Returns the total number of whole weeks in the duration.
    #[inline]
    pub fn num_weeks(&self) -> i32 {
        self.num_days() / 7
    }

    /// Returns the total number of whole days in the duration.
    pub fn num_days(&self) -> i32 {
        if self.days < 0 {
            let negated = -*self;
            -negated.days
        } else {
            self.days
        }
    }

    /// Returns the total number of whole hours in the duration.
    #[inline]
    pub fn num_hours(&self) -> i64 {
        self.num_seconds() / 3600
    }

    /// Returns the total number of whole minutes in the duration.
    #[inline]
    pub fn num_minutes(&self) -> i64 {
        self.num_seconds() / 60
    }

    /// Returns the total number of whole seconds in the duration.
    pub fn num_seconds(&self) -> i64 {
        // cannot overflow, 2^32 * 86400 < 2^64
        fn secs((days, secs, _): (i64, u32, u32)) -> i64 {
            days as i64 * SECS_PER_DAY as i64 + secs as i64
        }
        if self.days < 0 {-secs(self.to_negated_tuple_64())} else {secs(self.to_tuple_64())}
    }

    /// Returns the total number of whole milliseconds in the duration.
    pub fn num_milliseconds(&self) -> i64 {
        // cannot overflow, 2^32 * 86400 * 1000 < 2^64
        fn millis((days, secs, nanos): (i64, u32, u32)) -> i64 {
            static MILLIS_PER_SEC: i64 = 1_000;
            static NANOS_PER_MILLI: i64 = 1_000_000;
            (days as i64 * MILLIS_PER_SEC * SECS_PER_DAY as i64 +
             secs as i64 * MILLIS_PER_SEC +
             nanos as i64 / NANOS_PER_MILLI)
        }
        if self.days < 0 {-millis(self.to_negated_tuple_64())} else {millis(self.to_tuple_64())}
    }

    /// Returns the total number of whole microseconds in the duration,
    /// or `None` on the overflow (exceeding 2^63 microseconds in either directions).
    pub fn num_microseconds(&self) -> Option<i64> {
        fn micros((days, secs, nanos): (i64, u32, u32)) -> Option<i64> {
            static MICROS_PER_SEC: i64 = 1_000_000;
            static MICROS_PER_DAY: i64 = MICROS_PER_SEC * SECS_PER_DAY as i64;
            static NANOS_PER_MICRO: i64 = 1_000;
            let nmicros = try_opt!((days as i64).checked_mul(&MICROS_PER_DAY));
            let nmicros = try_opt!(nmicros.checked_add(&(secs as i64 * MICROS_PER_SEC)));
            let nmicros = try_opt!(nmicros.checked_add(&(nanos as i64 / NANOS_PER_MICRO as i64)));
            Some(nmicros)
        }
        if self.days < 0 {
            // the final negation won't overflow since we start with positive numbers.
            micros(self.to_negated_tuple_64()).map(|micros| -micros)
        } else {
            micros(self.to_tuple_64())
        }
    }

    /// Returns the total number of whole nanoseconds in the duration,
    /// or `None` on the overflow (exceeding 2^63 nanoseconds in either directions).
    pub fn num_nanoseconds(&self) -> Option<i64> {
        fn nanos((days, secs, nanos): (i64, u32, u32)) -> Option<i64> {
            static NANOS_PER_DAY: i64 = NANOS_PER_SEC as i64 * SECS_PER_DAY as i64;
            let nnanos = try_opt!((days as i64).checked_mul(&NANOS_PER_DAY));
            let nnanos = try_opt!(nnanos.checked_add(&(secs as i64 * NANOS_PER_SEC as i64)));
            let nnanos = try_opt!(nnanos.checked_add(&(nanos as i64)));
            Some(nnanos)
        }
        if self.days < 0 {
            // the final negation won't overflow since we start with positive numbers.
            nanos(self.to_negated_tuple_64()).map(|micros| -micros)
        } else {
            nanos(self.to_tuple_64())
        }
    }
}

impl num::Bounded for Duration {
    #[inline] fn min_value() -> Duration { MIN }
    #[inline] fn max_value() -> Duration { MAX }
}

impl num::Zero for Duration {
    #[inline]
    fn zero() -> Duration {
        Duration { days: 0, secs: 0, nanos: 0 }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.days == 0 && self.secs == 0 && self.nanos == 0
    }
}

impl Neg<Duration> for Duration {
    #[inline]
    fn neg(&self) -> Duration {
        let (days, secs, nanos) = self.to_negated_tuple_64();
        Duration { days: days as i32, secs: secs, nanos: nanos } // FIXME can overflow
    }
}

impl Add<Duration,Duration> for Duration {
    fn add(&self, rhs: &Duration) -> Duration {
        let mut days = self.days + rhs.days;
        let mut secs = self.secs + rhs.secs;
        let mut nanos = self.nanos + rhs.nanos;
        if nanos >= NANOS_PER_SEC as u32 {
            nanos -= NANOS_PER_SEC as u32;
            secs += 1;
        }
        if secs >= SECS_PER_DAY as u32 {
            secs -= SECS_PER_DAY as u32;
            days += 1;
        }
        Duration { days: days, secs: secs, nanos: nanos }
    }
}

impl num::CheckedAdd for Duration {
    fn checked_add(&self, rhs: &Duration) -> Option<Duration> {
        let mut days = try_opt!(self.days.checked_add(&rhs.days));
        let mut secs = self.secs + rhs.secs;
        let mut nanos = self.nanos + rhs.nanos;
        if nanos >= NANOS_PER_SEC as u32 {
            nanos -= NANOS_PER_SEC as u32;
            secs += 1;
        }
        if secs >= SECS_PER_DAY as u32 {
            secs -= SECS_PER_DAY as u32;
            days = try_opt!(days.checked_add(&1));
        }
        Some(Duration { days: days, secs: secs, nanos: nanos })
    }
}

impl Sub<Duration,Duration> for Duration {
    fn sub(&self, rhs: &Duration) -> Duration {
        let mut days = self.days - rhs.days;
        let mut secs = self.secs as i32 - rhs.secs as i32;
        let mut nanos = self.nanos as i32 - rhs.nanos as i32;
        if nanos < 0 {
            nanos += NANOS_PER_SEC;
            secs -= 1;
        }
        if secs < 0 {
            secs += SECS_PER_DAY;
            days -= 1;
        }
        Duration { days: days, secs: secs as u32, nanos: nanos as u32 }
    }
}

impl num::CheckedSub for Duration {
    fn checked_sub(&self, rhs: &Duration) -> Option<Duration> {
        let mut days = try_opt!(self.days.checked_sub(&rhs.days));
        let mut secs = self.secs as i32 - rhs.secs as i32;
        let mut nanos = self.nanos as i32 - rhs.nanos as i32;
        if nanos < 0 {
            nanos += NANOS_PER_SEC;
            secs -= 1;
        }
        if secs < 0 {
            secs += SECS_PER_DAY;
            days = try_opt!(days.checked_sub(&1));
        }
        Some(Duration { days: days, secs: secs as u32, nanos: nanos as u32 })
    }
}

impl Mul<i32,Duration> for Duration {
    fn mul(&self, rhs: &i32) -> Duration {
        /// Given `0 <= y < limit <= 2^30`,
        /// returns `(h,l)` such that `x * y = h * limit + l` where `0 <= l < limit`.
        fn mul_i64_u32_limit(x: i64, y: u32, limit: u32) -> (i64,u32) {
            let y = y as i64;
            let limit = limit as i64;
            let (xh, xl) = div_mod_floor_64(x, limit);
            let (h, l) = (xh * y, xl * y);
            let (h_, l) = div_rem_64(l, limit);
            (h + h_, l as u32)
        }

        let rhs = *rhs as i64;
        let (secs1, nanos) = mul_i64_u32_limit(rhs, self.nanos, NANOS_PER_SEC as u32);
        let (days1, secs1) = div_mod_floor_64(secs1, (SECS_PER_DAY as i64));
        let (days2, secs2) = mul_i64_u32_limit(rhs, self.secs, SECS_PER_DAY as u32);
        let mut days = self.days as i64 * rhs + days1 + days2;
        let mut secs = secs1 as u32 + secs2;
        if secs >= SECS_PER_DAY as u32 {
            secs -= 1;
            days += 1;
        }
        Duration { days: days as i32, secs: secs, nanos: nanos }
    }
}

impl Div<i32,Duration> for Duration {
    fn div(&self, rhs: &i32) -> Duration {
        let (rhs, days, secs, nanos) = if *rhs < 0 {
            let (days, secs, nanos) = self.to_negated_tuple_64();
            (-(*rhs as i64), days, secs as i64, nanos as i64)
        } else {
            (*rhs as i64, self.days as i64, self.secs as i64, self.nanos as i64)
        };

        let (days, carry) = div_mod_floor_64(days, rhs);
        let secs = secs + carry * SECS_PER_DAY as i64;
        let (secs, carry) = div_mod_floor_64(secs, rhs);
        let nanos = nanos + carry * NANOS_PER_SEC as i64;
        let nanos = nanos / rhs;
        Duration { days: days as i32, secs: secs as u32, nanos: nanos as u32 }
    }
}

impl fmt::Show for Duration {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let hasdate = self.days != 0;
        let hastime = (self.secs != 0 || self.nanos != 0) || !hasdate;

        try!(write!(f, "P"));
        if hasdate {
            // technically speaking the negative part is not the valid ISO 8601,
            // but we need to print it anyway.
            try!(write!(f, "{}D", self.days));
        }
        if hastime {
            if self.nanos == 0 {
                try!(write!(f, "T{}S", self.secs));
            } else if self.nanos % 1_000_000 == 0 {
                try!(write!(f, "T{}.{:03}S", self.secs, self.nanos / 1_000_000));
            } else if self.nanos % 1_000 == 0 {
                try!(write!(f, "T{}.{:06}S", self.secs, self.nanos / 1_000));
            } else {
                try!(write!(f, "T{}.{:09}S", self.secs, self.nanos));
            }
        }
        Ok(())
    }
}

// Copied from libnum
#[inline]
fn div_mod_floor(this: i32, other: i32) -> (i32, i32) {
    (div_floor(this, other), mod_floor(this, other))
}

#[inline]
fn div_floor(this: i32, other: i32) -> i32 {
    match div_rem(this, other) {
        (d, r) if (r > 0 && other < 0)
               || (r < 0 && other > 0) => d - 1,
        (d, _)                         => d,
    }
}

#[inline]
fn mod_floor(this: i32, other: i32) -> i32 {
    match this % other {
        r if (r > 0 && other < 0)
          || (r < 0 && other > 0) => r + other,
        r                         => r,
    }
}

#[inline]
fn div_rem(this: i32, other: i32) -> (i32, i32) {
    (this / other, this % other)
}

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
    use super::{Duration, MIN_DAYS, MAX_DAYS, MIN, MAX};
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
        assert_eq!(Duration::days(2) + Duration::seconds(86399) + Duration::nanoseconds(1234567890),
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
        assert_eq!(Duration::days(i32::MAX).num_days(), i32::MAX);
        assert_eq!(Duration::days(i32::MIN).num_days(), i32::MIN);
        assert_eq!(MAX.num_days(), MAX_DAYS);
        assert_eq!(MIN.num_days(), MIN_DAYS);
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
        assert_eq!(Duration::seconds(i32::MAX).num_seconds(), i32::MAX as i64);
        assert_eq!(Duration::seconds(i32::MIN).num_seconds(), i32::MIN as i64);
        assert_eq!(MAX.num_seconds(), (MAX_DAYS as i64 + 1) * 86400 - 1);
        assert_eq!(MIN.num_seconds(), MIN_DAYS as i64 * 86400);
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
        assert_eq!(Duration::milliseconds(i32::MAX).num_milliseconds(), i32::MAX as i64);
        assert_eq!(Duration::milliseconds(i32::MIN).num_milliseconds(), i32::MIN as i64);
        assert_eq!(MAX.num_milliseconds(), (MAX_DAYS as i64 + 1) * 86400_000 - 1);
        assert_eq!(MIN.num_milliseconds(), MIN_DAYS as i64 * 86400_000);
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
        assert_eq!(Duration::microseconds(i32::MAX).num_microseconds(), Some(i32::MAX as i64));
        assert_eq!(Duration::microseconds(i32::MIN).num_microseconds(), Some(i32::MIN as i64));
        assert_eq!(MAX.num_microseconds(), None);
        assert_eq!(MIN.num_microseconds(), None);

        // overflow checks
        static MICROS_PER_DAY: i64 = 86400_000_000;
        assert_eq!(Duration::days((i64::MAX / MICROS_PER_DAY) as i32).num_microseconds(),
                   Some(i64::MAX / MICROS_PER_DAY * MICROS_PER_DAY));
        assert_eq!(Duration::days((i64::MIN / MICROS_PER_DAY) as i32).num_microseconds(),
                   Some(i64::MIN / MICROS_PER_DAY * MICROS_PER_DAY));
        assert_eq!(Duration::days((i64::MAX / MICROS_PER_DAY + 1) as i32).num_microseconds(), None);
        assert_eq!(Duration::days((i64::MIN / MICROS_PER_DAY - 1) as i32).num_microseconds(), None);
    }

    #[test]
    fn test_duration_num_nanoseconds() {
        let d: Duration = Zero::zero();
        assert_eq!(d.num_nanoseconds(), Some(0));
        assert_eq!(Duration::nanoseconds(1).num_nanoseconds(), Some(1));
        assert_eq!(Duration::nanoseconds(-1).num_nanoseconds(), Some(-1));
        assert_eq!(Duration::nanoseconds(i32::MAX).num_nanoseconds(), Some(i32::MAX as i64));
        assert_eq!(Duration::nanoseconds(i32::MIN).num_nanoseconds(), Some(i32::MIN as i64));
        assert_eq!(MAX.num_nanoseconds(), None);
        assert_eq!(MIN.num_nanoseconds(), None);

        // overflow checks
        static NANOS_PER_DAY: i64 = 86400_000_000_000;
        assert_eq!(Duration::days((i64::MAX / NANOS_PER_DAY) as i32).num_nanoseconds(),
                   Some(i64::MAX / NANOS_PER_DAY * NANOS_PER_DAY));
        assert_eq!(Duration::days((i64::MIN / NANOS_PER_DAY) as i32).num_nanoseconds(),
                   Some(i64::MIN / NANOS_PER_DAY * NANOS_PER_DAY));
        assert_eq!(Duration::days((i64::MAX / NANOS_PER_DAY + 1) as i32).num_nanoseconds(), None);
        assert_eq!(Duration::days((i64::MIN / NANOS_PER_DAY - 1) as i32).num_nanoseconds(), None);
    }

    #[test]
    fn test_duration_checked_ops() {
        assert_eq!(Duration::days(MAX_DAYS).checked_add(&Duration::seconds(86399)),
                   Some(Duration::days(MAX_DAYS - 1) + Duration::seconds(86400+86399)));
        assert!(Duration::days(MAX_DAYS).checked_add(&Duration::seconds(86400)).is_none());

        assert_eq!(Duration::days(MIN_DAYS).checked_sub(&Duration::seconds(0)),
                   Some(Duration::days(MIN_DAYS)));
        assert!(Duration::days(MIN_DAYS).checked_sub(&Duration::seconds(1)).is_none());
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
