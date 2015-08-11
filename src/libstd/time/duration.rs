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

#![unstable(feature = "duration", reason = "recently added API per RFC 1040")]

#[cfg(stage0)]
use prelude::v1::*;

use cmp;
use fmt;
use io::{self, Cursor, SeekFrom};
use io::prelude::*;
use ops::{Add, Sub, Mul, Div};
use str;
use sys::time::SteadyTime;

const NANOS_PER_SEC: u32 = 1_000_000_000;
const NANOS_PER_MILLI: u32 = 1_000_000;
const MILLIS_PER_SEC: u64 = 1_000;
const NANOS_PER_MICRO: u32 = 1_000;

/// A duration type to represent a span of time, typically used for system
/// timeouts.
///
/// Each duration is composed of a number of seconds and nanosecond precision.
/// APIs binding a system timeout will typically round up the nanosecond
/// precision if the underlying system does not support that level of precision.
///
/// Durations implement many common traits, including `Add`, `Sub`, and other
/// ops traits. Currently a duration may only be inspected for its number of
/// seconds and its nanosecond precision.
///
/// # Examples
///
/// ```
/// #![feature(duration)]
/// use std::time::Duration;
///
/// let five_seconds = Duration::new(5, 0);
/// let five_seconds_and_five_nanos = five_seconds + Duration::new(0, 5);
///
/// assert_eq!(five_seconds_and_five_nanos.secs(), 5);
/// assert_eq!(five_seconds_and_five_nanos.extra_nanos(), 5);
///
/// let ten_millis = Duration::from_millis(10);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Duration {
    secs: u64,
    nanos: u32, // Always 0 <= nanos < NANOS_PER_SEC
}

impl Duration {
    /// Crates a new `Duration` from the specified number of seconds and
    /// additional nanosecond precision.
    ///
    /// If the nanoseconds is greater than 1 billion (the number of nanoseconds
    /// in a second), then it will carry over into the seconds provided.
    pub fn new(secs: u64, nanos: u32) -> Duration {
        let secs = secs + (nanos / NANOS_PER_SEC) as u64;
        let nanos = nanos % NANOS_PER_SEC;
        Duration { secs: secs, nanos: nanos }
    }

    /// Runs a closure, returning the duration of time it took to run the
    /// closure.
    #[unstable(feature = "duration_span",
               reason = "unsure if this is the right API or whether it should \
                         wait for a more general \"moment in time\" \
                         abstraction")]
    pub fn span<F>(f: F) -> Duration where F: FnOnce() {
        let start = SteadyTime::now();
        f();
        &SteadyTime::now() - &start
    }

    /// Creates a new `Duration` from the specified number of seconds.
    pub fn from_secs(secs: u64) -> Duration {
        Duration { secs: secs, nanos: 0 }
    }

    /// Creates a new `Duration` from the specified number of milliseconds.
    pub fn from_millis(millis: u64) -> Duration {
        let secs = millis / MILLIS_PER_SEC;
        let nanos = ((millis % MILLIS_PER_SEC) as u32) * NANOS_PER_MILLI;
        Duration { secs: secs, nanos: nanos }
    }

    /// Returns the number of whole seconds represented by this duration.
    ///
    /// The extra precision represented by this duration is ignored (e.g. extra
    /// nanoseconds are not represented in the returned value).
    pub fn secs(&self) -> u64 { self.secs }

    /// Returns the nanosecond precision represented by this duration.
    ///
    /// This method does **not** return the length of the duration when
    /// represented by nanoseconds. The returned number always represents a
    /// fractional portion of a second (e.g. it is less than one billion).
    pub fn extra_nanos(&self) -> u32 { self.nanos }
}

impl Add for Duration {
    type Output = Duration;

    fn add(self, rhs: Duration) -> Duration {
        let mut secs = self.secs.checked_add(rhs.secs)
                           .expect("overflow when adding durations");
        let mut nanos = self.nanos + rhs.nanos;
        if nanos >= NANOS_PER_SEC {
            nanos -= NANOS_PER_SEC;
            secs = secs.checked_add(1).expect("overflow when adding durations");
        }
        debug_assert!(nanos < NANOS_PER_SEC);
        Duration { secs: secs, nanos: nanos }
    }
}

impl Sub for Duration {
    type Output = Duration;

    fn sub(self, rhs: Duration) -> Duration {
        let mut secs = self.secs.checked_sub(rhs.secs)
                           .expect("overflow when subtracting durations");
        let nanos = if self.nanos >= rhs.nanos {
            self.nanos - rhs.nanos
        } else {
            secs = secs.checked_sub(1)
                       .expect("overflow when subtracting durations");
            self.nanos + NANOS_PER_SEC - rhs.nanos
        };
        debug_assert!(nanos < NANOS_PER_SEC);
        Duration { secs: secs, nanos: nanos }
    }
}

impl Mul<u32> for Duration {
    type Output = Duration;

    fn mul(self, rhs: u32) -> Duration {
        // Multiply nanoseconds as u64, because it cannot overflow that way.
        let total_nanos = self.nanos as u64 * rhs as u64;
        let extra_secs = total_nanos / (NANOS_PER_SEC as u64);
        let nanos = (total_nanos % (NANOS_PER_SEC as u64)) as u32;
        let secs = self.secs.checked_mul(rhs as u64)
                       .and_then(|s| s.checked_add(extra_secs))
                       .expect("overflow when multiplying duration");
        debug_assert!(nanos < NANOS_PER_SEC);
        Duration { secs: secs, nanos: nanos }
    }
}

impl Div<u32> for Duration {
    type Output = Duration;

    fn div(self, rhs: u32) -> Duration {
        let secs = self.secs / (rhs as u64);
        let carry = self.secs - secs * (rhs as u64);
        let extra_nanos = carry * (NANOS_PER_SEC as u64) / (rhs as u64);
        let nanos = self.nanos / rhs + (extra_nanos as u32);
        debug_assert!(nanos < NANOS_PER_SEC);
        Duration { secs: secs, nanos: nanos }
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // FIXME: fmt::Formatter::with_padding() isn't public, and the pad() function applies
        // precision in a way we don't want. At such time as it becomes reasonable to support
        // padding without reimplementing fmt::Formatter::with_padding() here, this should be
        // updated to support padding. In preparation for that, we're already writing to a
        // stack buffer.

        // µ (U+00B5) is 0xC2B5 in UTF-8.
        // The following is >= the longest possible string we can format.
        let mut buf = *b"18446744073709551615.999999999\xC2\xB5s";

        fn write_to_buf<'a>(dur: &Duration, f: &fmt::Formatter, mut buf: &'a mut [u8])
                           -> io::Result<&'a str> {
            fn backup_over_zeros<'a>(cursor: &mut Cursor<&'a mut [u8]>) {
                let mut pos = cursor.position() as usize;
                {
                    let buf = cursor.get_ref();
                    loop {
                        match buf[pos-1] {
                            b'0' => { pos -= 1; }
                            b'.' => { pos -= 1; break; }
                            _    => { break; }
                        }
                    }
                }
                cursor.set_position(pos as u64);
            }

            let mut cursor = Cursor::new(&mut buf[..]);
            let precision = f.precision();
            match (dur.secs, dur.nanos) {
                (0, n) if n < NANOS_PER_MICRO => try!(write!(cursor, "{}ns", n)),
                (s, 0) if precision.unwrap_or(0) == 0 => try!(write!(cursor, "{}s", s)),
                (0, n) => {
                    let (unit, suffix, max_prec) = if n >= NANOS_PER_MILLI {
                        (NANOS_PER_MILLI, "ms", 6)
                    } else {
                        (NANOS_PER_MICRO, "µs", 3)
                    };
                    // Leverage our existing floating-point formatting implementation
                    let n = n as f64 / unit as f64;
                    if let Some(precision) = precision {
                        try!(write!(cursor, "{:.*}{}", cmp::min(precision, max_prec), n, suffix));
                    } else {
                        // variable precision up to 3 decimals
                        // just print all 3 decimals and then back up over zeroes.
                        try!(write!(cursor, "{:.3}", n));
                        backup_over_zeros(&mut cursor);
                        try!(write!(cursor, "{}", suffix));
                    }
                }
                (s, n) => {
                    try!(write!(cursor, "{}", s));
                    // we're going to cheat a little here and re-use the same float trick above.
                    // but because 0.1234 has a leading 0, we're going to back up a byte, save it,
                    // overwrite it, and then restore it.
                    let n = n as f64 / NANOS_PER_SEC as f64;
                    try!(cursor.seek(SeekFrom::Current(-1)));
                    let saved_pos = cursor.position() as usize;
                    let saved_digit = cursor.get_ref()[saved_pos];
                    if let Some(precision) = precision {
                        try!(write!(cursor, "{:.*}s", cmp::min(precision, 9), n));
                    } else {
                        // variable precision up to 3 decimals
                        try!(write!(cursor, "{:.3}", n));
                        backup_over_zeros(&mut cursor);
                        try!(write!(cursor, "s"));
                    }
                    // make sure we didn't round up to 1.0 when printing the float
                    if cursor.get_ref()[saved_pos] == b'1' {
                        // we did. back up and rewrite the seconds
                        if s == u64::max_value() {
                            // we can't represent a larger value. Luckily, we know that
                            // u64::max_value() ends with '5', so we can just replace it with '6'.
                            cursor.get_mut()[saved_pos] = b'6';
                        } else {
                            try!(cursor.seek(SeekFrom::Start(0)));
                            try!(write!(cursor, "{}", s+1));
                            match precision {
                                None | Some(0) => {},
                                Some(precision) => {
                                    // we need to write out the trailing zeroes
                                    try!(write!(cursor, "."));
                                    for _ in 0..cmp::min(precision, 9) {
                                        try!(write!(cursor, "0"));
                                    }
                                }
                            }
                            try!(write!(cursor, "s"));
                        }
                    } else {
                        // restore the digit we overwrite earlier
                        cursor.get_mut()[saved_pos] = saved_digit;
                    }
                }
            }
            // buf now contains our fully-formatted value
            let pos = cursor.position() as usize;
            let buf = &cursor.into_inner()[..pos];
            // formatting always writes strings
            Ok(unsafe { str::from_utf8_unchecked(buf) })
        }

        let result = write_to_buf(self, f, &mut buf);
        // writing to the stack buffer should never fail
        debug_assert!(result.is_ok(), "Error in <Duration as Display>::fmt: {:?}", result);
        let buf_str = try!(result.or(Err(fmt::Error)));
        // If fmt::Formatter::with_padding() was public, this is where we'd use it.
        write!(f, "{}", buf_str)
    }
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;
    use super::Duration;

    #[test]
    fn creation() {
        assert!(Duration::from_secs(1) != Duration::from_secs(0));
        assert_eq!(Duration::from_secs(1) + Duration::from_secs(2),
                   Duration::from_secs(3));
        assert_eq!(Duration::from_millis(10) + Duration::from_secs(4),
                   Duration::new(4, 10 * 1_000_000));
        assert_eq!(Duration::from_millis(4000), Duration::new(4, 0));
    }

    #[test]
    fn secs() {
        assert_eq!(Duration::new(0, 0).secs(), 0);
        assert_eq!(Duration::from_secs(1).secs(), 1);
        assert_eq!(Duration::from_millis(999).secs(), 0);
        assert_eq!(Duration::from_millis(1001).secs(), 1);
    }

    #[test]
    fn nanos() {
        assert_eq!(Duration::new(0, 0).extra_nanos(), 0);
        assert_eq!(Duration::new(0, 5).extra_nanos(), 5);
        assert_eq!(Duration::new(0, 1_000_000_001).extra_nanos(), 1);
        assert_eq!(Duration::from_secs(1).extra_nanos(), 0);
        assert_eq!(Duration::from_millis(999).extra_nanos(), 999 * 1_000_000);
        assert_eq!(Duration::from_millis(1001).extra_nanos(), 1 * 1_000_000);
    }

    #[test]
    fn add() {
        assert_eq!(Duration::new(0, 0) + Duration::new(0, 1),
                   Duration::new(0, 1));
        assert_eq!(Duration::new(0, 500_000_000) + Duration::new(0, 500_000_001),
                   Duration::new(1, 1));
    }

    #[test]
    fn sub() {
        assert_eq!(Duration::new(0, 1) - Duration::new(0, 0),
                   Duration::new(0, 1));
        assert_eq!(Duration::new(0, 500_000_001) - Duration::new(0, 500_000_000),
                   Duration::new(0, 1));
        assert_eq!(Duration::new(1, 0) - Duration::new(0, 1),
                   Duration::new(0, 999_999_999));
    }

    #[test] #[should_panic]
    fn sub_bad1() {
        Duration::new(0, 0) - Duration::new(0, 1);
    }

    #[test] #[should_panic]
    fn sub_bad2() {
        Duration::new(0, 0) - Duration::new(1, 0);
    }

    #[test]
    fn mul() {
        assert_eq!(Duration::new(0, 1) * 2, Duration::new(0, 2));
        assert_eq!(Duration::new(1, 1) * 3, Duration::new(3, 3));
        assert_eq!(Duration::new(0, 500_000_001) * 4, Duration::new(2, 4));
        assert_eq!(Duration::new(0, 500_000_001) * 4000,
                   Duration::new(2000, 4000));
    }

    #[test]
    fn div() {
        assert_eq!(Duration::new(0, 1) / 2, Duration::new(0, 0));
        assert_eq!(Duration::new(1, 1) / 3, Duration::new(0, 333_333_333));
        assert_eq!(Duration::new(99, 999_999_000) / 100,
                   Duration::new(0, 999_999_990));
    }

    #[test]
    fn display() {
        assert_eq!(Duration::new(0, 1).to_string(), "1ns");
        assert_eq!(Duration::new(0, 1_000).to_string(), "1µs");
        assert_eq!(Duration::new(0, 1_000_000).to_string(), "1ms");
        assert_eq!(Duration::new(1, 0).to_string(), "1s");
        assert_eq!(Duration::new(0, 999).to_string(), "999ns");
        assert_eq!(Duration::new(0, 1_001).to_string(), "1.001µs");
        assert_eq!(Duration::new(0, 1_100).to_string(), "1.1µs");
        assert_eq!(Duration::new(0, 1_234_567).to_string(), "1.235ms");
        assert_eq!(Duration::new(1, 234_567_890).to_string(), "1.235s");
        assert_eq!(Duration::new(1, 999_999).to_string(), "1.001s");

        assert_eq!(Duration::new(0, 2).to_string(), "2ns");
        assert_eq!(Duration::new(0, 2_000).to_string(), "2µs");
        assert_eq!(Duration::new(0, 2_000_000).to_string(), "2ms");
        assert_eq!(Duration::new(2, 0).to_string(), "2s");
        assert_eq!(Duration::new(2, 2).to_string(), "2s");
        assert_eq!(Duration::new(2, 2_000_000).to_string(), "2.002s");
        assert_eq!(Duration::new(0, 2_000_002).to_string(), "2ms");
        assert_eq!(Duration::new(2, 2_000_002).to_string(), "2.002s");
        assert_eq!(Duration::new(0, 0).to_string(), "0ns");

        assert_eq!(format!("{:.9}", Duration::new(2, 2)), "2.000000002s");
        assert_eq!(format!("{:.6}", Duration::new(2, 2)), "2.000000s");
        assert_eq!(format!("{:.6}", Duration::new(0, 2_000_002)), "2.000002ms");
        assert_eq!(format!("{:.20}", Duration::new(1, 1)), "1.000000001s");
        assert_eq!(format!("{:.3}", Duration::new(1, 0)), "1.000s");
        assert_eq!(format!("{:.3}", Duration::new(0, 1)), "1ns");
        assert_eq!(format!("{:.0}", Duration::new(0, 1_001)), "1µs");
        assert_eq!(format!("{:.0}", Duration::new(0, 1_500)), "2µs");
        assert_eq!(format!("{:.9}", Duration::new(0, 100)), "100ns");
        assert_eq!(format!("{:.9}", Duration::new(0, 100_000)), "100.000µs");
        assert_eq!(format!("{:.9}", Duration::new(0, 100_000_000)), "100.000000ms");
        assert_eq!(format!("{:.9}", Duration::new(100,0)), "100.000000000s");

        assert_eq!(format!("{:.9}", Duration::new(u64::max_value(), 999_999_999)),
                   "18446744073709551615.999999999s");
        assert_eq!(Duration::new(u64::max_value(), 999_999_999).to_string(),
                   "18446744073709551616s");
        assert_eq!(format!("{:.3}", Duration::new(u64::max_value(), 999_999_999)),
                   "18446744073709551616.000s")
    }
}
