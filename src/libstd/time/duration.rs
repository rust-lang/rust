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
use from_str::FromStr;
use ops::{Add, Sub, Mul, Div, Neg};
use option::{Option, Some, None};
use num;
use num::{CheckedAdd, CheckedMul};
use result::{Result, Ok, Err};

/// The number of milliseconds in a second.
const MILLIS_PER_SECOND: i64 = 1000;
/// The number of milliseconds in a minute.
const MILLIS_PER_MINUTE: i64 = 60 * MILLIS_PER_SECOND;
/// The number of milliseconds in an hour.
const MILLIS_PER_HOUR: i64 = 60 * MILLIS_PER_MINUTE;
/// The number of milliseconds in a day.
const MILLIS_PER_DAY: i64 = 24 * MILLIS_PER_HOUR;
/// The number of milliseconds in a week.
const MILLIS_PER_WEEK: i64 = 7 * MILLIS_PER_DAY;

/// The number of microseconds in a millisecond.
const MICROS_PER_MILLI: i64 = 1000;
/// The number of nanoseconds in a microsecond.
const NANOS_PER_MICRO: i64 = 1000;
/// The number of nanoseconds in a millisecond.
const NANOS_PER_MILLI: i64 = 1000_000;

const OUT_OF_BOUNDS: &'static str = "Duration out of bounds";

/// An absolute amount of time, independent of time zones and calendars with nanosecond precision.
/// A duration can express the positive or negative difference between two instants in time
/// according to a particular clock.
#[deriving(Clone, PartialEq, Eq, PartialOrd, Ord, Zero, Default, Hash)]
pub struct Duration {
    millis: i64, // Milliseconds
    nanos:  i32, // Nanoseconds, |nanos| < NANOS_PER_MILLI
}

macro_rules! try_opt(
    ($e:expr) => (match $e { Some(v) => v, None => return None })
)
macro_rules! duration(
    ($millis:expr, $nanos:expr) => (Duration { millis: $millis, nanos: $nanos })
)
macro_rules! carry(
    ($millis:expr, $nanos:expr) => (
        if $millis > 0 && $nanos < 0 {
            (-1, NANOS_PER_MILLI)
        }
        else if $millis < 0 && $nanos > 0 {
            (1, -NANOS_PER_MILLI)
        }
        else {
            (0, 0)
        }
    )
)

/// The minimum possible `Duration` (-P106751991167DT7H12M55.808999999S).
pub const MIN: Duration = duration!(i64::MIN, -NANOS_PER_MILLI as i32 + 1);
/// The maximum possible `Duration` (P106751991167DT7H12M55.807999999S).
pub const MAX: Duration = duration!(i64::MAX,  NANOS_PER_MILLI as i32 - 1);

impl Duration {
    /// Makes a new `Duration` with given number of weeks with overflow checks.
    /// Fails when the duration is out of bounds.
    #[inline]
    pub fn weeks(weeks: i64) -> Duration {
        duration!(weeks.checked_mul(&MILLIS_PER_WEEK).expect(OUT_OF_BOUNDS), 0)
    }

    /// Makes a new `Duration` with given number of days with overflow checks.
    /// Fails when the duration is out of bounds.
    #[inline]
    pub fn days(days: i64) -> Duration {
        duration!(days.checked_mul(&MILLIS_PER_DAY).expect(OUT_OF_BOUNDS), 0)
    }

    /// Makes a new `Duration` with given number of hours with overflow checks.
    /// Fails when the duration is out of bounds.
    #[inline]
    pub fn hours(hours: i64) -> Duration {
        duration!(hours.checked_mul(&MILLIS_PER_HOUR).expect(OUT_OF_BOUNDS), 0)
    }

    /// Makes a new `Duration` with given number of minutes with overflow checks.
    /// Fails when the duration is out of bounds.
    #[inline]
    pub fn minutes(minutes: i64) -> Duration {
        duration!(minutes.checked_mul(&MILLIS_PER_MINUTE).expect(OUT_OF_BOUNDS), 0)
    }

    /// Makes a new `Duration` with given number of seconds with overflow checks.
    /// Fails when the duration is out of bounds.
    #[inline]
    pub fn seconds(seconds: i64) -> Duration {
        duration!(seconds.checked_mul(&MILLIS_PER_SECOND).expect(OUT_OF_BOUNDS), 0)
    }

    /// Makes a new `Duration` with given number of milliseconds.
    #[inline]
    pub fn milliseconds(milliseconds: i64) -> Duration {
        duration!(milliseconds, 0)
    }

    /// Makes a new `Duration` with given number of microseconds.
    #[inline]
    pub fn microseconds(microseconds: i64) -> Duration {
        duration!(
            microseconds / MICROS_PER_MILLI,
            (NANOS_PER_MICRO * (microseconds % MICROS_PER_MILLI)) as i32
        )
    }

    /// Makes a new `Duration` with given number of nanoseconds.
    #[inline]
    pub fn nanoseconds(nanoseconds: i64) -> Duration {
        duration!(nanoseconds / NANOS_PER_MILLI, (nanoseconds % NANOS_PER_MILLI) as i32)
    }

    /// Runs a closure, returning the duration of time it took to run the
    /// closure.
    pub fn span(f: ||) -> Duration {
        let before = super::precise_time_ns();
        f();
        Duration::nanoseconds((before - super::precise_time_ns()) as i64)
    }

    /// Returns the total number of whole weeks in the duration.
    #[inline]
    pub fn num_weeks(&self) -> i64 {
        self.millis / MILLIS_PER_WEEK
    }

    /// Returns the total number of whole days in the duration.
    #[inline]
    pub fn num_days(&self) -> i64 {
        self.millis / MILLIS_PER_DAY
    }

    /// Returns the total number of whole hours in the duration.
    #[inline]
    pub fn num_hours(&self) -> i64 {
        self.millis / MILLIS_PER_HOUR
    }

    /// Returns the total number of whole minutes in the duration.
    #[inline]
    pub fn num_minutes(&self) -> i64 {
        self.millis / MILLIS_PER_MINUTE
    }

    /// Returns the total number of whole seconds in the duration.
    #[inline]
    pub fn num_seconds(&self) -> i64 {
        self.millis / MILLIS_PER_SECOND
    }

    /// Returns the total number of milliseconds in the duration.
    #[inline]
    pub fn num_milliseconds(&self) -> i64 {
        self.millis
    }

    /// Returns the total number of microseconds in the duration.
    /// or `None` on overflow (exceeding 2^63 microseconds in either direction).
    #[inline]
    pub fn num_microseconds(&self) -> Option<i64> {
        let micros = try_opt!(self.millis.checked_mul(&MICROS_PER_MILLI));
        micros.checked_add(&(self.nanos as i64/ NANOS_PER_MICRO))
    }

    /// Returns the total number of nanoseconds in the duration.
    /// or `None` on overflow (exceeding 2^63 nanoseconds in either direction).
    #[inline]
    pub fn num_nanoseconds(&self) -> Option<i64> {
        let nanos = try_opt!(self.millis.checked_mul(&NANOS_PER_MILLI));
        nanos.checked_add(&(self.nanos as i64))
    }
}

impl num::Bounded for Duration {
    #[inline] fn min_value() -> Duration { MIN }
    #[inline] fn max_value() -> Duration { MAX }
}

impl Neg<Duration> for Duration {
    #[inline]
    fn neg(&self) -> Duration {
        duration!(-self.millis, -self.nanos)
    }
}

impl Add<Duration,Duration> for Duration {
    fn add(&self, rhs: &Duration) -> Duration {
        let nanos_sum = self.nanos + rhs.nanos;

        let millis = self.millis + rhs.millis + (nanos_sum as i64) / NANOS_PER_MILLI;
        let nanos = nanos_sum % (NANOS_PER_MILLI as i32);

        let (m, n) = carry!(millis, nanos);

        duration!(millis + m, nanos + n as i32)
    }
}

impl Sub<Duration,Duration> for Duration {
    fn sub(&self, rhs: &Duration) -> Duration {
        let nanos_sub = self.nanos - rhs.nanos;

        let millis = self.millis - rhs.millis + (nanos_sub as i64) / NANOS_PER_MILLI;
        let nanos = nanos_sub % (NANOS_PER_MILLI as i32);

        let (m, n) = carry!(millis, nanos);

        duration!(millis + m, nanos + n as i32)
    }
}

impl num::CheckedAdd for Duration {
    fn checked_add(&self, rhs: &Duration) -> Option<Duration> {
        let nanos_sum = self.nanos + rhs.nanos; // cannot overflow
        let millis_sum = try_opt!(self.millis.checked_add(&rhs.millis));

        let millis = try_opt!(millis_sum.checked_add(&((nanos_sum as i64) / NANOS_PER_MILLI)));
        let nanos = nanos_sum % (NANOS_PER_MILLI as i32);

        let (m, n) = carry!(millis, nanos);

        Some(duration!(try_opt!(millis.checked_add(&m)), nanos + n as i32))
    }
}


impl num::CheckedSub for Duration {
    fn checked_sub(&self, rhs: &Duration) -> Option<Duration> {
        let nanos_sub = self.nanos - rhs.nanos; // cannot overflow
        let millis_sub = try_opt!(self.millis.checked_sub(&rhs.millis));

        let millis = try_opt!(millis_sub.checked_sub(&((nanos_sub as i64) / NANOS_PER_MILLI)));
        let nanos = nanos_sub % (NANOS_PER_MILLI as i32);

        let (m, n) = carry!(millis, nanos);

        Some(duration!(try_opt!(millis.checked_add(&m)), nanos + n as i32))
    }
}

impl Mul<i32,Duration> for Duration {
    fn mul(&self, rhs: &i32) -> Duration {
        let nanos_mul = self.nanos as i64 * *rhs as i64;

        let millis = self.millis * *rhs as i64 + nanos_mul / NANOS_PER_MILLI;
        let nanos = (nanos_mul % NANOS_PER_MILLI) as i32;

        duration!(millis, nanos)
    }
}

impl Div<i32,Duration> for Duration {
    fn div(&self, rhs: &i32) -> Duration {
        let millis = self.millis / *rhs as i64;
        let nanos = ((self.millis % *rhs as i64)*NANOS_PER_MILLI + self.nanos as i64)/ *rhs as i64;

        duration!(millis + nanos / NANOS_PER_MILLI, (nanos % NANOS_PER_MILLI) as i32)
    }
}

impl fmt::Show for Duration {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{}P", if self.millis < 0 || self.nanos < 0 { "-" } else { "" }));

        let days = self.num_days();
        let mut rem = (self.millis - days * MILLIS_PER_DAY).abs();

        let hours = rem / MILLIS_PER_HOUR;
        rem -= hours * MILLIS_PER_HOUR;

        let minutes = rem / MILLIS_PER_MINUTE;
        rem -= minutes * MILLIS_PER_MINUTE;

        let seconds = rem / MILLIS_PER_SECOND;
        rem = (rem - seconds * MILLIS_PER_SECOND) * NANOS_PER_MILLI + (self.nanos as i64).abs();

        let hasdate = days != 0;
        let hastime = (hours != 0 || minutes != 0 || seconds != 0 || rem != 0) || !hasdate;

        if hasdate {
            try!(write!(f, "{}D", days.abs()));
        }

        if hastime {
            try!(write!(f, "T"));

            if hours != 0 {
                try!(write!(f, "{}H", hours));
            }

            if minutes != 0 {
                try!(write!(f, "{}M", minutes));
            }

            if rem == 0 {
                try!(write!(f, "{}S", seconds));
            }
            else if rem % NANOS_PER_MILLI == 0 {
                try!(write!(f, "{}.{:03}S", seconds, rem / NANOS_PER_MILLI));
            }
            else if rem % NANOS_PER_MICRO == 0 {
                try!(write!(f, "{}.{:06}S", seconds, rem / NANOS_PER_MICRO));
            }
            else {
                try!(write!(f, "{}.{:09}S", seconds, rem));
            }
        }

        Ok(())
    }
}

impl FromStr for Duration {
    /// Parse a `Duration` from a string.
    ///
    /// Durations are represented by the format ±P[n]DT[n]H[n]M[n]S, where [n] is a decimal positive
    /// number (last one possibly using a comma), and the remaining characters are case-insensitive.
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!(from_str::<Duration>("-P106751991167DT7H12M55.808999999S"), Some(Duration::MIN));
    /// assert_eq!(from_str::<Duration>("P106751991167DT7H12M55.807999999S"), Some(Duration::MAX));
    /// assert_eq!(from_str::<Duration>("P31D"), Some(Duration::days(31)));
    /// assert_eq!(from_str::<Duration>("not even a Duration"), None);
    /// ```
    #[inline]
    fn from_str(source: &str) -> Option<Duration> {
        fn atoi(source: &str) -> (Option<i64>, i64, &str) {
            let mut value: Option<i64> = None;
            let mut scale = 1i64;
            let mut s = source;

            while !s.is_empty() {
                s = match s.slice_shift_char() {
                    (Some(c), rem) if '0' <= c && c <= '9' => {
                        let digit = (c as u8 - b'0') as i64;

                        value = value.unwrap_or(0).checked_mul(&10);

                        if value.is_none() {
                            break;
                        }

                        value = value.unwrap().checked_add(&digit);

                        if value.is_none() {
                            break;
                        }

                        scale *= 10;
                        rem
                    },
                    _ => break
                }
            }

            (value, scale, s)
        }

        let source = source.trim_left();

        // Sign
        let (mut s, sign) = match source.slice_shift_char() {
            (Some(c), rem) if c == '-' || c == '+' => (rem, c),
            _ => (source, '+'),
        };

        // P - period
        s = match s.slice_shift_char() {
            (Some(c), rem) if c == 'P' || c == 'p' => rem,
            _ => return None
        };

        // D - days
        let (days, _, next) = atoi(s);

        if days.is_some() {
            s = match next.slice_shift_char() {
                (Some(c), rem) if c == 'D' || c == 'd' => rem,
                _ => return None
            };
        }

        // T - time
        s = match s.slice_shift_char() {
            (Some(c), rem) if c == 'T' || c == 't' => rem,
            (None, _) if days.is_some() => s,
            _ => return None
        };

        let (mut hours, mut minutes): (Option<i64>, Option<i64>) = (None, None);
        let (mut seconds, mut nanos): (Option<i64>, Option<i64>) = (None, None);

        while !s.is_empty() {
            let (value, scale, next) = atoi(s);

            if value.is_none() || next.is_empty() {
                return None;
            }

            s = match next.slice_shift_char() {
                (Some(c), rem) if (c == 'H' || c == 'h')
                                            && hours.is_none() && minutes.is_none()
                                            && seconds.is_none() && nanos.is_none() => {
                    hours = value;
                    rem
                },
                (Some(c), rem) if (c == 'M' || c == 'm')
                                            && minutes.is_none() && seconds.is_none()
                                            && nanos.is_none() => {
                    minutes = value;
                    rem
                },
                (Some(c), rem) if (c == '.') && seconds.is_none() && nanos.is_none() => {
                    seconds = value;
                    rem
                },
                (Some(c), rem) if (c == 'S' || c == 's')
                                            && (seconds.is_none() || nanos.is_none()) => {
                    if seconds.is_none() {
                        seconds = value;
                    }
                    else {
                        nanos = value.unwrap().checked_mul(&(MILLIS_PER_SECOND * NANOS_PER_MILLI));

                        if nanos.is_none() {
                            return None;
                        }

                        nanos = Some(nanos.unwrap() / scale);
                    }

                    rem
                },
                _ => return None
            }
        };

        let days_millis = try_opt!(days.unwrap_or(0).checked_mul(&MILLIS_PER_DAY));
        let hours_millis = try_opt!(hours.unwrap_or(0).checked_mul(&MILLIS_PER_HOUR));
        let minutes_millis = try_opt!(minutes.unwrap_or(0).checked_mul(&MILLIS_PER_MINUTE));
        let seconds_millis = try_opt!(seconds.unwrap_or(0).checked_mul(&MILLIS_PER_SECOND));

        let mut millis = try_opt!(days_millis.checked_add(&hours_millis));

        millis = try_opt!(millis.checked_add(&minutes_millis));
        millis = try_opt!(millis.checked_add(&seconds_millis));

        let mut nanos = nanos.unwrap_or(0);

        if sign == '-' {
            millis = -millis;
            nanos = -nanos;
        }

        Some(
            duration!(
                try_opt!(millis.checked_add(&(nanos / NANOS_PER_MILLI))),
                (nanos % NANOS_PER_MILLI) as i32
            )
        )
    }
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
        assert_eq!(Duration::days(-42).to_string(), "-P42D".to_string());
        assert_eq!(Duration::seconds(42).to_string(), "PT42S".to_string());
        assert_eq!(Duration::milliseconds(42).to_string(), "PT0.042S".to_string());
        assert_eq!(Duration::microseconds(42).to_string(), "PT0.000042S".to_string());
        assert_eq!(Duration::nanoseconds(42).to_string(), "PT0.000000042S".to_string());
        assert_eq!((Duration::days(7) + Duration::milliseconds(6543)).to_string(),
                   "P7DT6.543S".to_string());
        assert_eq!(Duration::seconds(-86401).to_string(), "-P1DT1S".to_string());
        assert_eq!(Duration::nanoseconds(-1).to_string(), "-PT0.000000001S".to_string());
        assert_eq!(MIN.to_string(), "-P106751991167DT7H12M55.808999999S".to_string());
        assert_eq!(MAX.to_string(), "P106751991167DT7H12M55.807999999S".to_string());

        // the format specifier should have no effect on `Duration`
        assert_eq!(format!("{:30}", Duration::days(1) + Duration::milliseconds(2345)),
                   "P1DT2.345S".to_string());
    }

    #[test]
    fn test_duration_from_str() {
        assert_eq!(from_str::<Duration>("not even a Duration"), None);
        assert_eq!(from_str::<Duration>("DT755S"), None);
        assert_eq!(from_str::<Duration>("P123T9S"), None);
        assert_eq!(from_str::<Duration>("PDT7S"), None);
        assert_eq!(from_str::<Duration>("P1D7S"), None);
        assert_eq!(from_str::<Duration>("P1D"), Some(Duration::days(1)));
        assert_eq!(from_str::<Duration>("+P2D"), Some(Duration::days(2)));
        assert_eq!(from_str::<Duration>("-P3D"), Some(Duration::days(-3)));
        assert_eq!(from_str::<Duration>("-PT5H"), Some(Duration::hours(-5)));
        assert_eq!(from_str::<Duration>("PT5H30M"), Some(Duration::seconds(19800)));
        assert_eq!(from_str::<Duration>("PT30M"), Some(Duration::minutes(30)));
        assert_eq!(from_str::<Duration>("PT30M5H"), None);
        assert_eq!(from_str::<Duration>("PT1M5S"), Some(Duration::seconds(65)));
        assert_eq!(from_str::<Duration>("PT5S1M"), None);
        assert_eq!(from_str::<Duration>("PT5S1H"), None);
        assert_eq!(from_str::<Duration>("PT1H2M5S"), Some(Duration::seconds(3725)));
        assert_eq!(from_str::<Duration>("PT1H1H2M5S"), None);
        assert_eq!(from_str::<Duration>("PT1H2M1H5S"), None);
        assert_eq!(from_str::<Duration>("PT1H2M5S1H"), None);
        assert_eq!(from_str::<Duration>("PT1H2M5.S"), None);
        assert_eq!(from_str::<Duration>("PT1H2M5.5S"), Some(Duration::milliseconds(3725500)));
        assert_eq!(from_str::<Duration>("PT1H2M5.05S"), Some(Duration::milliseconds(3725050)));
        assert_eq!(from_str::<Duration>("PT1H2M5.005S"), Some(Duration::milliseconds(3725005)));
        assert_eq!(from_str::<Duration>("PT1H2M5.0005S"), Some(duration!(3725000, 500000)));
        assert_eq!(from_str::<Duration>("PT1H2M5.00005S"), Some(duration!(3725000, 50000)));
        assert_eq!(from_str::<Duration>("PT1H2M5.000005S"), Some(duration!(3725000, 5000)));
        assert_eq!(from_str::<Duration>("PT1H2M5.0000005S"), Some(duration!(3725000, 500)));
        assert_eq!(from_str::<Duration>("PT1H2M5.00000005S"), Some(duration!(3725000, 50)));
        assert_eq!(from_str::<Duration>("PT1H2M5.000000005S"), Some(duration!(3725000, 5)));
        assert_eq!(from_str::<Duration>("PT1H2M5.500000005S"), Some(duration!(3725500, 5)));
        assert_eq!(from_str::<Duration>("-P106751991167DT7H12M55.808999999S"), Some(MIN));
        assert_eq!(from_str::<Duration>("P106751991167DT7H12M55.807999999S"), Some(MAX));
    }
}

