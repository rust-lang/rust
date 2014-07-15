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
#![allow(missing_doc)] // FIXME

use {fmt, num, i32};
use option::{Option, Some, None};
use result::{Result, Ok, Err};
use ops::{Neg, Add, Sub, Mul, Div};
use num::{CheckedAdd, ToPrimitive};

pub static MIN_DAYS: i32 = i32::MIN;
pub static MAX_DAYS: i32 = i32::MAX;

static NANOS_PER_SEC: i32 = 1_000_000_000;
static SECS_PER_DAY: i32 = 86400;

macro_rules! earlyexit(
    ($e:expr) => (match $e { Some(v) => v, None => return None })
)

/// ISO 8601 duration
#[deriving(PartialEq, Eq, PartialOrd, Ord)]
pub struct Duration {
    days: i32,
    secs: u32,
    nanos: u32,
}

impl Duration {
    pub fn new(days: i32, secs: i32, nanos: i32) -> Option<Duration> {
        let (secs_, nanos) = div_mod_floor(nanos, NANOS_PER_SEC);
        let secs = earlyexit!(secs.checked_add(&secs_));
        let (days_, secs) = div_mod_floor(secs, SECS_PER_DAY);
        let days = earlyexit!(days.checked_add(&days_).and_then(|v| v.to_i32()));
        Some(Duration { days: days, secs: secs as u32, nanos: nanos as u32 })
    }

    #[inline]
    pub fn weeks(weeks: i32) -> Duration {
        Duration::days(weeks * 7)
    }

    #[inline]
    pub fn days(days: i32) -> Duration {
        let days = days.to_i32().expect("Duration::days out of bounds");
        Duration { days: days, secs: 0, nanos: 0 }
    }

    #[inline]
    pub fn hours(hours: i32) -> Duration {
        let (days, hours) = div_mod_floor(hours, (SECS_PER_DAY / 3600));
        let secs = hours * 3600;
        Duration { secs: secs as u32, ..Duration::days(days) }
    }

    #[inline]
    pub fn minutes(mins: i32) -> Duration {
        let (days, mins) = div_mod_floor(mins, (SECS_PER_DAY / 60));
        let secs = mins * 60;
        Duration { secs: secs as u32, ..Duration::days(days) }
    }

    #[inline]
    pub fn seconds(secs: i32) -> Duration {
        let (days, secs) = div_mod_floor(secs, SECS_PER_DAY);
        Duration { secs: secs as u32, ..Duration::days(days) }
    }

    #[inline]
    pub fn milliseconds(millis: i32) -> Duration {
        let (secs, millis) = div_mod_floor(millis, (NANOS_PER_SEC / 1_000_000));
        let nanos = millis * 1_000_000;
        Duration { nanos: nanos as u32, ..Duration::seconds(secs) }
    }

    #[inline]
    pub fn microseconds(micros: i32) -> Duration {
        let (secs, micros) = div_mod_floor(micros, (NANOS_PER_SEC / 1_000));
        let nanos = micros * 1_000;
        Duration { nanos: nanos as u32, ..Duration::seconds(secs) }
    }

    #[inline]
    pub fn nanoseconds(nanos: i32) -> Duration {
        let (secs, nanos) = div_mod_floor(nanos, NANOS_PER_SEC);
        Duration { nanos: nanos as u32, ..Duration::seconds(secs) }
    }

    #[inline]
    pub fn ndays(&self) -> i32 {
        self.days as i32
    }

    #[inline]
    pub fn nseconds(&self) -> u32 {
        self.secs as u32
    }

    #[inline]
    pub fn nnanoseconds(&self) -> u32 {
        self.nanos as u32
    }
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
    fn neg(&self) -> Duration {
        // FIXME overflow (e.g. `-Duration::days(i32::MIN as i32)`)
        let mut days = -(self.days as i32);
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
        Duration { days: days as i32, secs: secs as u32, nanos: nanos as u32 }
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
        let mut days = earlyexit!(self.days.checked_add(&rhs.days));
        let mut secs = self.secs + rhs.secs;
        let mut nanos = self.nanos + rhs.nanos;
        if nanos >= NANOS_PER_SEC as u32 {
            nanos -= NANOS_PER_SEC as u32;
            secs += 1;
        }
        if secs >= SECS_PER_DAY as u32 {
            secs -= SECS_PER_DAY as u32;
            days = earlyexit!(days.checked_add(&1));
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
        let mut days = earlyexit!(self.days.checked_sub(&rhs.days));
        let mut secs = self.secs as i32 - rhs.secs as i32;
        let mut nanos = self.nanos as i32 - rhs.nanos as i32;
        if nanos < 0 {
            nanos += NANOS_PER_SEC;
            secs -= 1;
        }
        if secs < 0 {
            secs += SECS_PER_DAY;
            days = earlyexit!(days.checked_sub(&1));
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
            let negated = -*self;
            (-*rhs as i64, negated.days as i64, negated.secs as i64, negated.nanos as i64)
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

        try!('P'.fmt(f));
        if hasdate {
            // technically speaking the negative part is not the valid ISO 8601,
            // but we need to print it anyway.
            try!(write!(f, "{}D", self.days));
        }
        if hastime {
            if self.nanos == 0 {
                try!(write!(f, "T{}S", self.secs));
            } else if self.nanos % 1_000_000 == 0 {
                try!(write!(f, "T{},{:03}S", self.secs, self.nanos / 1_000_000));
            } else if self.nanos % 1_000 == 0 {
                try!(write!(f, "T{},{:06}S", self.secs, self.nanos / 1_000));
            } else {
                try!(write!(f, "T{},{:09}S", self.secs, self.nanos));
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
    use option::Some;
    use super::{Duration, MIN_DAYS, MAX_DAYS};
    use i32;
    use num::{CheckedAdd, CheckedSub, Zero};
    use to_string::ToString;

    fn zero() -> Duration { Zero::zero() }

    #[test]
    fn test_duration() {
        assert!(zero() != Duration::seconds(1));
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
        assert_eq!(zero() * i32::MAX, zero());
        assert_eq!(zero() * i32::MIN, zero());
        assert_eq!(Duration::nanoseconds(1) * 0, zero());
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
        assert_eq!(zero() / i32::MAX, zero());
        assert_eq!(zero() / i32::MIN, zero());
        assert_eq!(Duration::nanoseconds(123_456_789) / 1, Duration::nanoseconds(123_456_789));
        assert_eq!(Duration::nanoseconds(123_456_789) / -1, -Duration::nanoseconds(123_456_789));
        assert_eq!(-Duration::nanoseconds(123_456_789) / -1, Duration::nanoseconds(123_456_789));
        assert_eq!(-Duration::nanoseconds(123_456_789) / 1, -Duration::nanoseconds(123_456_789));
    }

    #[test]
    fn test_duration_fmt() {
        assert_eq!(zero().to_string(), "PT0S".to_string());
        assert_eq!(Duration::days(42).to_string(), "P42D".to_string());
        assert_eq!(Duration::days(-42).to_string(), "P-42D".to_string());
        assert_eq!(Duration::seconds(42).to_string(), "PT42S".to_string());
        assert_eq!(Duration::milliseconds(42).to_string(), "PT0,042S".to_string());
        assert_eq!(Duration::microseconds(42).to_string(), "PT0,000042S".to_string());
        assert_eq!(Duration::nanoseconds(42).to_string(), "PT0,000000042S".to_string());
        assert_eq!((Duration::days(7) + Duration::milliseconds(6543)).to_string(),
                   "P7DT6,543S".to_string());
    }
}
