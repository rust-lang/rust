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
use num::{Bounded, CheckedAdd, CheckedSub};
use ops::{Add, Sub, Mul, Div, Neg};
use option::{Option, Some, None};

macro_rules! try_opt(
    ($e:expr) => (match $e { Some(v) => v, None => return None })
)

/// An absolute amount of time, independent of time zones and calendars with tick precision.
/// A single tick represents 100 nanoseconds. A duration can express the positive or negative
/// difference between two instants in time according to a particular clock.
#[deriving(Clone, PartialEq, Eq, PartialOrd, Ord, Zero, Default, Hash, Rand)]
pub struct Duration(pub i64);

/// The minimum possible `Duration` (-P10675199DT2H48M5.4775808S).
pub const MIN: Duration = Duration(i64::MIN);
/// The maximum possible `Duration` (P10675199DT2H48M5.4775807S).
pub const MAX: Duration = Duration(i64::MAX);

/// The number of ticks in a microsecond.
pub const TICKS_PER_MICROSECOND: i64 = 10;
/// The number of ticks in a millisecond.
pub const TICKS_PER_MILLISECOND: i64 = 1000 * TICKS_PER_MICROSECOND;
/// The number of ticks in a second.
pub const TICKS_PER_SECOND: i64 = 1000 * TICKS_PER_MILLISECOND;
/// The number of ticks in a minute.
pub const TICKS_PER_MINUTE: i64 = 60 * TICKS_PER_SECOND;
/// The number of ticks in an hour.
pub const TICKS_PER_HOUR: i64 = 60 * TICKS_PER_MINUTE;
/// The number of ticks in a day.
pub const TICKS_PER_DAY: i64 = 24 * TICKS_PER_HOUR;

const OUT_OF_BOUNDS: &'static str = "Duration out of bounds";

impl Duration {
    /// Makes a new `Duration` with given number of microseconds.
    /// Equivalent to Duration(n * TICKS_PER_MICROSECOND) with overflow checks.
    /// Fails when the duration is out of bounds.
    #[inline]
    pub fn microseconds(n: i64) -> Duration {
		let ticks = n.checked_mul(&TICKS_PER_MICROSECOND).expect(OUT_OF_BOUNDS);
 		Duration(ticks)
	}

    /// Makes a new `Duration` with given number of milliseconds.
    /// Equivalent to Duration(n * TICKS_PER_MILLISECOND) with overflow checks.
    /// Fails when the duration is out of bounds.
    #[inline]
    pub fn milliseconds(n: i64) -> Duration {
        let ticks = n.checked_mul(&TICKS_PER_MILLISECOND).expect(OUT_OF_BOUNDS);
		Duration(ticks)
    }

    /// Makes a new `Duration` with given number of seconds.
    /// Equivalent to Duration(n * TICKS_PER_SECOND) with overflow checks.
    /// Fails when the duration is out of bounds.
    #[inline]
    pub fn seconds(n: i64) -> Duration {
        let ticks = n.checked_mul(&TICKS_PER_SECOND).expect(OUT_OF_BOUNDS);
		Duration(ticks)
    }

    /// Makes a new `Duration` with given number of minutes.
    /// Equivalent to Duration(n * TICKS_PER_MINUTE) with overflow checks.
    /// Fails when the duration is out of bounds.
    #[inline]
    pub fn minutes(n: i64) -> Duration {
        let ticks = n.checked_mul(&TICKS_PER_MINUTE).expect(OUT_OF_BOUNDS);
		Duration(ticks)
    }

    /// Makes a new `Duration` with given number of hours.
    /// Equivalent to Duration(n * TICKS_PER_HOUR) with overflow checks.
    /// Fails when the duration is out of bounds.
    #[inline]
    pub fn hours(n: i64) -> Duration {
        let ticks = n.checked_mul(&TICKS_PER_HOUR).expect(OUT_OF_BOUNDS);
		Duration(ticks)
    }

    /// Makes a new `Duration` with given number of days.
    /// Equivalent to Duration(n * TICKS_PER_DAY) with overflow checks.
    /// Fails when the duration is out of bounds.
    #[inline]
    pub fn days(n: i64) -> Duration {
        let ticks = n.checked_mul(&TICKS_PER_DAY).expect(OUT_OF_BOUNDS);
		Duration(ticks)
    }

    /// Returns the total number of whole days in the duration.
    #[inline]
    pub fn num_days(&self) -> i64 {
		self.num_ticks() / TICKS_PER_DAY
    }

    /// Returns the total number of whole hours in the duration.
    #[inline]
    pub fn num_hours(&self) -> i64 {
        self.num_ticks() / TICKS_PER_HOUR
    }

    /// Returns the total number of whole minutes in the duration.
    #[inline]
    pub fn num_minutes(&self) -> i64 {
        self.num_ticks() / TICKS_PER_MINUTE
    }

    /// Returns the total number of whole seconds in the duration.
    #[inline]
    pub fn num_seconds(&self) -> i64 {
        self.num_ticks() / TICKS_PER_SECOND
    }

    /// Returns the total number of whole milliseconds in the duration.
    #[inline]
    pub fn num_milliseconds(&self) -> i64 {
        self.num_ticks() / TICKS_PER_MILLISECOND
    }

    /// Returns the total number of whole microseconds in the duration.
    #[inline]
    pub fn num_microseconds(&self) -> i64 {
        self.num_ticks() / TICKS_PER_MICROSECOND
    }

    /// Returns the total number of ticks in the duration.
    #[inline]
    pub fn num_ticks(&self) -> i64 {
        let &Duration(ticks) = self;
        ticks
    }
}

impl Bounded for Duration {
    #[inline] fn min_value() -> Duration { MIN }
    #[inline] fn max_value() -> Duration { MAX }
}

impl Neg<Duration> for Duration {
    #[inline]
    fn neg(&self) -> Duration {
        Duration(-self.num_ticks())
    }
}

impl Add<Duration, Duration> for Duration {
    fn add(&self, rhs: &Duration) -> Duration {
        Duration(self.num_ticks() + rhs.num_ticks())
    }
}

impl CheckedAdd for Duration {
    fn checked_add(&self, rhs: &Duration) -> Option<Duration> {
		let result = try_opt!(self.num_ticks().checked_add(&rhs.num_ticks()));
		Some(Duration(result))
    }
}

impl Sub<Duration,Duration> for Duration {
    fn sub(&self, rhs: &Duration) -> Duration {
        Duration(self.num_ticks() - rhs.num_ticks())
    }
}

impl CheckedSub for Duration {
    fn checked_sub(&self, rhs: &Duration) -> Option<Duration> {
		let result = try_opt!(self.num_ticks().checked_sub(&rhs.num_ticks()));
        Some(Duration(result))
    }
}

impl Mul<i64, Duration> for Duration {
    fn mul(&self, rhs: &i64) -> Duration {
        Duration(*rhs * self.num_ticks())
    }
}

impl Div<i64, Duration> for Duration {
    fn div(&self, rhs: &i64) -> Duration {
        Duration(self.num_ticks() / *rhs)
    }
}

impl fmt::Show for Duration {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ticks = self.num_ticks();

        try!(write!(f, "{}P", if ticks < 0 { "-" } else { "" }));

        let days = ticks / TICKS_PER_DAY;
		ticks = (ticks - days * TICKS_PER_DAY).abs();

        let hours = ticks / TICKS_PER_HOUR;
		ticks -= hours * TICKS_PER_HOUR;

        let minutes = ticks / TICKS_PER_MINUTE;
		ticks -= minutes * TICKS_PER_MINUTE;

        let seconds = ticks / TICKS_PER_SECOND;
		ticks -= seconds * TICKS_PER_SECOND;

        let hasdate = days != 0;
        let hastime = (hours != 0 || minutes != 0 || seconds != 0 || ticks != 0) || !hasdate;

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

			if ticks == 0 {
			    try!(write!(f, "{}S", seconds));
			}
			else if ticks % TICKS_PER_MILLISECOND == 0 {
			    try!(write!(f, "{}.{:03}S", seconds, ticks / TICKS_PER_MILLISECOND));
			}
			else if ticks % TICKS_PER_MICROSECOND == 0 {
			    try!(write!(f, "{}.{:06}S", seconds, ticks / TICKS_PER_MICROSECOND));
			}
			else {
			    try!(write!(f, "{}.{:07}S", seconds, ticks));
			}
		}

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Duration, MIN, MAX, TICKS_PER_MILLISECOND, TICKS_PER_MICROSECOND, TICKS_PER_SECOND};
    use {i32, i64};
    use num::{Zero};
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
        assert_eq!(-Duration::days(3), Duration::days(-3));
        assert_eq!(Duration::days(-3), -Duration::days(3));
        assert_eq!(-(Duration::days(1)*3 + Duration::seconds(70)),
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
        assert_eq!(Duration::days(10675199i64).num_days(), MAX.num_days());
        assert_eq!(Duration::days(-10675199i64).num_days(), MIN.num_days());
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
        assert_eq!(Duration::milliseconds(i32::MAX as i64).num_milliseconds(), i32::MAX as i64);
        assert_eq!(Duration::milliseconds(i32::MIN as i64).num_milliseconds(), i32::MIN as i64);
        assert_eq!(MAX.num_milliseconds(), i64::MAX/TICKS_PER_MILLISECOND);
        assert_eq!(MIN.num_milliseconds(), i64::MIN/TICKS_PER_MILLISECOND);
    }

    #[test]
    fn test_duration_num_microseconds() {
        let d: Duration = Zero::zero();
        assert_eq!(d.num_microseconds(), 0);
        assert_eq!(Duration::microseconds(1).num_microseconds(), 1);
        assert_eq!(Duration::microseconds(-1).num_microseconds(), -1);
        assert_eq!(Duration(999).num_microseconds(),   99);
        assert_eq!(Duration(1001).num_microseconds(),  100);
        assert_eq!(Duration(-999).num_microseconds(), -99);
        assert_eq!(Duration(-1001).num_microseconds(),-100);
        assert_eq!(MAX.num_microseconds(), i64::MAX/TICKS_PER_MICROSECOND);
        assert_eq!(MIN.num_microseconds(), i64::MIN/TICKS_PER_MICROSECOND);
        assert_eq!(Duration::microseconds(1) * 10i64, Duration::microseconds(10));
        assert_eq!(Duration::microseconds(1) * 10, Duration::microseconds(10));
        assert_eq!(Duration::microseconds(12) / 4, Duration::microseconds(3));
        // overflow checks
        assert_eq!(Duration::microseconds(i64::MAX/TICKS_PER_MICROSECOND)
                    .checked_add(&Duration::seconds(1)), None);
        assert_eq!(Duration::microseconds(101).checked_sub(&Duration::microseconds(1)), 
                    Some(Duration::microseconds(100)));
        assert_eq!(Duration::microseconds(0).checked_sub(&Duration::microseconds(1)), 
                    Some(Duration::microseconds(-1)));
    }

    #[test]
    fn test_duration_num_ticks() {
        assert_eq!(Duration(i64::MAX), MAX);
        assert_eq!(Duration(i64::MIN), MIN);
        assert_eq!(Duration(i64::MAX / 2) * 2, MAX - Duration(1));
        assert_eq!(Duration(i64::MIN / 2) * 2, MIN);

        // overflow checks
        assert_eq!(Duration(i64::MIN + 1).checked_sub(&Duration(1)), Some(MIN));
        assert_eq!(Duration(i64::MIN).checked_sub(&Duration(1)), None);
        assert_eq!(Duration(i64::MAX - 1).checked_add(&Duration(1)), Some(MAX));
        assert_eq!(Duration(i64::MAX).checked_add(&Duration(1)), None);
    }

    #[test]
    fn test_duration_mul() {
        let d: Duration = Zero::zero();
        assert_eq!(d * i64::MAX, d);
        assert_eq!(d * i64::MIN, d);
        assert_eq!(Duration(1) * 0, Zero::zero());
        assert_eq!(Duration(1) * 1, Duration(1));
        assert_eq!(Duration(1) *  TICKS_PER_SECOND,  Duration::seconds(1));
        assert_eq!(Duration(1) * -TICKS_PER_SECOND, -Duration::seconds(1));
        assert_eq!(-Duration(1) * TICKS_PER_SECOND, -Duration::seconds(1));
        assert_eq!((Duration(1) + Duration::seconds(1) + Duration::days(1)) * 3,
                   Duration(3) + Duration::seconds(3) + Duration::days(3));
        assert_eq!(Duration::milliseconds(1500) * -2, Duration::seconds(-3));
        assert_eq!(Duration::milliseconds(-1500) * 2, Duration::seconds(-3));
    }

    #[test]
    fn test_duration_div() {
        let d: Duration = Zero::zero();
        assert_eq!(d / i64::MAX, d);
        assert_eq!(d / i64::MIN, d);
        assert_eq!(Duration(123_456_789)  /  1,  Duration(123_456_789));
        assert_eq!(Duration(123_456_789)  / -1, -Duration(123_456_789));
        assert_eq!(-Duration(123_456_789) / -1,  Duration(123_456_789));
        assert_eq!(-Duration(123_456_789) /  1, -Duration(123_456_789));
        assert_eq!(Duration::seconds(1) / 3, Duration(TICKS_PER_SECOND/3));
        assert_eq!(Duration::seconds(4) / 3, Duration(4*TICKS_PER_SECOND/3));
        assert_eq!(Duration::seconds(-1) / 2, Duration::milliseconds(-500));
        assert_eq!(Duration::seconds(1) / -2, Duration::milliseconds(-500));
        assert_eq!(Duration::seconds(-1) / -2, Duration::milliseconds(500));
        assert_eq!(Duration::seconds(-4) / 3, Duration(-4*TICKS_PER_SECOND/3));
        assert_eq!(Duration::seconds(-4) / -3, Duration(4*TICKS_PER_SECOND/3));
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
        assert_eq!(Duration(42).to_string(), "PT0.0000042S".to_string());
        assert_eq!((Duration::days(7) + Duration::milliseconds(6543)).to_string(),
                   "P7DT6.543S".to_string());
        assert_eq!(Duration::seconds(-86401).to_string(), "-P1DT1S".to_string());
        assert_eq!(Duration(-1).to_string(), "-PT0.0000001S".to_string());
        assert_eq!(MIN.to_string(), "-P10675199DT2H48M5.4775808S".to_string());
        assert_eq!((MIN + Duration(1)).to_string(), "-P10675199DT2H48M5.4775807S".to_string());
        assert_eq!(MAX.to_string(), "P10675199DT2H48M5.4775807S".to_string());
        assert_eq!((MAX - Duration(1)).to_string(), "P10675199DT2H48M5.4775806S".to_string());

        // the format specifier should have no effect on `Duration`
        assert_eq!(format!("{:30}", Duration::days(1) + Duration::milliseconds(2345)),
                   "P1DT2.345S".to_string());
    }
}
