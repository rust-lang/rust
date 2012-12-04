// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[forbid(deprecated_mode)];

use core::cmp::Eq;
use libc::{c_char, c_int, c_long, size_t, time_t};
use io::{Reader, ReaderUtil};
use result::{Result, Ok, Err};

#[abi = "cdecl"]
extern mod rustrt {
    #[legacy_exports]
    fn get_time(sec: &mut i64, nsec: &mut i32);

    fn precise_time_ns(ns: &mut u64);

    fn rust_tzset();
    // FIXME: The i64 values can be passed by-val when #2064 is fixed.
    fn rust_gmtime(&&sec: i64, &&nsec: i32, &&result: Tm);
    fn rust_localtime(&&sec: i64, &&nsec: i32, &&result: Tm);
    fn rust_timegm(&&tm: Tm, sec: &mut i64);
    fn rust_mktime(&&tm: Tm, sec: &mut i64);
}

/// A record specifying a time value in seconds and nanoseconds.
#[auto_serialize]
#[auto_deserialize]
pub struct Timespec { sec: i64, nsec: i32 }

impl Timespec {
    static fn new(sec: i64, nsec: i32) -> Timespec {
        Timespec { sec: sec, nsec: nsec }
    }
}

impl Timespec : Eq {
    pure fn eq(&self, other: &Timespec) -> bool {
        self.sec == other.sec && self.nsec == other.nsec
    }
    pure fn ne(&self, other: &Timespec) -> bool { !self.eq(other) }
}

/**
 * Returns the current time as a `timespec` containing the seconds and
 * nanoseconds since 1970-01-01T00:00:00Z.
 */
pub fn get_time() -> Timespec {
    let mut sec = 0i64;
    let mut nsec = 0i32;
    rustrt::get_time(&mut sec, &mut nsec);
    return Timespec::new(sec, nsec);
}


/**
 * Returns the current value of a high-resolution performance counter
 * in nanoseconds since an unspecified epoch.
 */
pub fn precise_time_ns() -> u64 {
    let mut ns = 0u64;
    rustrt::precise_time_ns(&mut ns);
    ns
}


/**
 * Returns the current value of a high-resolution performance counter
 * in seconds since an unspecified epoch.
 */
pub fn precise_time_s() -> float {
    return (precise_time_ns() as float) / 1000000000.;
}

pub fn tzset() {
    rustrt::rust_tzset();
}

#[auto_serialize]
#[auto_deserialize]
pub struct Tm {
    tm_sec: i32, // seconds after the minute ~[0-60]
    tm_min: i32, // minutes after the hour ~[0-59]
    tm_hour: i32, // hours after midnight ~[0-23]
    tm_mday: i32, // days of the month ~[1-31]
    tm_mon: i32, // months since January ~[0-11]
    tm_year: i32, // years since 1900
    tm_wday: i32, // days since Sunday ~[0-6]
    tm_yday: i32, // days since January 1 ~[0-365]
    tm_isdst: i32, // Daylight Savings Time flag
    tm_gmtoff: i32, // offset from UTC in seconds
    tm_zone: ~str, // timezone abbreviation
    tm_nsec: i32, // nanoseconds
}

impl Tm : Eq {
    pure fn eq(&self, other: &Tm) -> bool {
        self.tm_sec == (*other).tm_sec &&
        self.tm_min == (*other).tm_min &&
        self.tm_hour == (*other).tm_hour &&
        self.tm_mday == (*other).tm_mday &&
        self.tm_mon == (*other).tm_mon &&
        self.tm_year == (*other).tm_year &&
        self.tm_wday == (*other).tm_wday &&
        self.tm_yday == (*other).tm_yday &&
        self.tm_isdst == (*other).tm_isdst &&
        self.tm_gmtoff == (*other).tm_gmtoff &&
        self.tm_zone == (*other).tm_zone &&
        self.tm_nsec == (*other).tm_nsec
    }
    pure fn ne(&self, other: &Tm) -> bool { !self.eq(other) }
}

pub pure fn empty_tm() -> Tm {
    Tm {
        tm_sec: 0_i32,
        tm_min: 0_i32,
        tm_hour: 0_i32,
        tm_mday: 0_i32,
        tm_mon: 0_i32,
        tm_year: 0_i32,
        tm_wday: 0_i32,
        tm_yday: 0_i32,
        tm_isdst: 0_i32,
        tm_gmtoff: 0_i32,
        tm_zone: ~"",
        tm_nsec: 0_i32,
    }
}

/// Returns the specified time in UTC
pub fn at_utc(clock: Timespec) -> Tm {
    let mut Timespec { sec, nsec } = clock;
    let mut tm = empty_tm();
    rustrt::rust_gmtime(sec, nsec, tm);
    move tm
}

/// Returns the current time in UTC
pub fn now_utc() -> Tm {
    at_utc(get_time())
}

/// Returns the specified time in the local timezone
pub fn at(clock: Timespec) -> Tm {
    let mut Timespec { sec, nsec } = clock;
    let mut tm = empty_tm();
    rustrt::rust_localtime(sec, nsec, tm);
    move tm
}

/// Returns the current time in the local timezone
pub fn now() -> Tm {
    at(get_time())
}

/// Parses the time from the string according to the format string.
pub pure fn strptime(s: &str, format: &str) -> Result<Tm, ~str> {
    // unsafe only because do_strptime is annoying to make pure
    // (it does IO with a str_reader)
    unsafe {do_strptime(s, format)}
}

/// Formats the time according to the format string.
pub pure fn strftime(format: &str, tm: &Tm) -> ~str {
    // unsafe only because do_strftime is annoying to make pure
    // (it does IO with a str_reader)
    move unsafe { do_strftime(format, tm) }
}

impl Tm {
    /// Convert time to the seconds from January 1, 1970
    fn to_timespec() -> Timespec {
        let mut sec = 0i64;
        if self.tm_gmtoff == 0_i32 {
            rustrt::rust_timegm(self, &mut sec);
        } else {
            rustrt::rust_mktime(self, &mut sec);
        }
        Timespec::new(sec, self.tm_nsec)
    }

    /// Convert time to the local timezone
    fn to_local() -> Tm {
        at(self.to_timespec())
    }

    /// Convert time to the UTC
    fn to_utc() -> Tm {
        at_utc(self.to_timespec())
    }

    /**
     * Return a string of the current time in the form
     * "Thu Jan  1 00:00:00 1970".
     */
    pure fn ctime() -> ~str { self.strftime(~"%c") }

    /// Formats the time according to the format string.
    pure fn strftime(&self, format: &str) -> ~str {
        move strftime(format, self)
    }

    /**
     * Returns a time string formatted according to RFC 822.
     *
     * local: "Thu, 22 Mar 2012 07:53:18 PST"
     * utc:   "Thu, 22 Mar 2012 14:53:18 UTC"
     */
    pure fn rfc822() -> ~str {
        if self.tm_gmtoff == 0_i32 {
            self.strftime(~"%a, %d %b %Y %T GMT")
        } else {
            self.strftime(~"%a, %d %b %Y %T %Z")
        }
    }

    /**
     * Returns a time string formatted according to RFC 822 with Zulu time.
     *
     * local: "Thu, 22 Mar 2012 07:53:18 -0700"
     * utc:   "Thu, 22 Mar 2012 14:53:18 -0000"
     */
    pure fn rfc822z() -> ~str {
        self.strftime(~"%a, %d %b %Y %T %z")
    }

    /**
     * Returns a time string formatted according to ISO 8601.
     *
     * local: "2012-02-22T07:53:18-07:00"
     * utc:   "2012-02-22T14:53:18Z"
     */
    pure fn rfc3339() -> ~str {
        if self.tm_gmtoff == 0_i32 {
            self.strftime(~"%Y-%m-%dT%H:%M:%SZ")
        } else {
            let s = self.strftime(~"%Y-%m-%dT%H:%M:%S");
            let sign = if self.tm_gmtoff > 0_i32 { '+' } else { '-' };
            let mut m = i32::abs(self.tm_gmtoff) / 60_i32;
            let h = m / 60_i32;
            m -= h * 60_i32;
            s + fmt!("%c%02d:%02d", sign, h as int, m as int)
        }
    }
}

priv fn do_strptime(s: &str, format: &str) -> Result<Tm, ~str> {
    fn match_str(s: &str, pos: uint, needle: &str) -> bool {
        let mut i = pos;
        for str::each(needle) |ch| {
            if s[i] != ch {
                return false;
            }
            i += 1u;
        }
        return true;
    }

    fn match_strs(ss: &str, pos: uint, strs: &[(~str, i32)])
      -> Option<(i32, uint)> {
        let mut i = 0u;
        let len = strs.len();
        while i < len {
            let &(needle, value) = &strs[i];

            if match_str(ss, pos, needle) {
                return Some((value, pos + str::len(needle)));
            }
            i += 1u;
        }

        None
    }

    fn match_digits(ss: &str, pos: uint, digits: uint, ws: bool)
      -> Option<(i32, uint)> {
        let mut pos = pos;
        let mut value = 0_i32;

        let mut i = 0u;
        while i < digits {
            let range = str::char_range_at(str::from_slice(ss), pos);
            pos = range.next;

            match range.ch {
              '0' .. '9' => {
                value = value * 10_i32 + (range.ch as i32 - '0' as i32);
              }
              ' ' if ws => (),
              _ => return None
            }
            i += 1u;
        }

        Some((value, pos))
    }

    fn parse_char(s: &str, pos: uint, c: char) -> Result<uint, ~str> {
        let range = str::char_range_at(s, pos);

        if c == range.ch {
            Ok(range.next)
        } else {
            Err(fmt!("Expected %?, found %?",
                str::from_char(c),
                str::from_char(range.ch)))
        }
    }

    fn parse_type(s: &str, pos: uint, ch: char, tm: &mut Tm)
      -> Result<uint, ~str> {
        match ch {
          'A' => match match_strs(s, pos, ~[
              (~"Sunday", 0_i32),
              (~"Monday", 1_i32),
              (~"Tuesday", 2_i32),
              (~"Wednesday", 3_i32),
              (~"Thursday", 4_i32),
              (~"Friday", 5_i32),
              (~"Saturday", 6_i32)
          ]) {
            Some(item) => { let (v, pos) = item; tm.tm_wday = v; Ok(pos) }
            None => Err(~"Invalid day")
          },
          'a' => match match_strs(s, pos, ~[
              (~"Sun", 0_i32),
              (~"Mon", 1_i32),
              (~"Tue", 2_i32),
              (~"Wed", 3_i32),
              (~"Thu", 4_i32),
              (~"Fri", 5_i32),
              (~"Sat", 6_i32)
          ]) {
            Some(item) => { let (v, pos) = item; tm.tm_wday = v; Ok(pos) }
            None => Err(~"Invalid day")
          },
          'B' => match match_strs(s, pos, ~[
              (~"January", 0_i32),
              (~"February", 1_i32),
              (~"March", 2_i32),
              (~"April", 3_i32),
              (~"May", 4_i32),
              (~"June", 5_i32),
              (~"July", 6_i32),
              (~"August", 7_i32),
              (~"September", 8_i32),
              (~"October", 9_i32),
              (~"November", 10_i32),
              (~"December", 11_i32)
          ]) {
            Some(item) => { let (v, pos) = item; tm.tm_mon = v; Ok(pos) }
            None => Err(~"Invalid month")
          },
          'b' | 'h' => match match_strs(s, pos, ~[
              (~"Jan", 0_i32),
              (~"Feb", 1_i32),
              (~"Mar", 2_i32),
              (~"Apr", 3_i32),
              (~"May", 4_i32),
              (~"Jun", 5_i32),
              (~"Jul", 6_i32),
              (~"Aug", 7_i32),
              (~"Sep", 8_i32),
              (~"Oct", 9_i32),
              (~"Nov", 10_i32),
              (~"Dec", 11_i32)
          ]) {
            Some(item) => { let (v, pos) = item; tm.tm_mon = v; Ok(pos) }
            None => Err(~"Invalid month")
          },
          'C' => match match_digits(s, pos, 2u, false) {
            Some(item) => {
                let (v, pos) = item;
                  tm.tm_year += (v * 100_i32) - 1900_i32;
                  Ok(pos)
              }
            None => Err(~"Invalid year")
          },
          'c' => {
                // FIXME(#3724): cleanup
                result::chain(
                result::chain(
                result::chain(
                result::chain(
                result::chain(
                result::chain(
                result::chain(
                result::chain(
                    move parse_type(s, pos, 'a', tm),
                    |pos| parse_char(s, pos, ' ')),
                    |pos| parse_type(s, pos, 'b', tm)),
                    |pos| parse_char(s, pos, ' ')),
                    |pos| parse_type(s, pos, 'e', tm)),
                    |pos| parse_char(s, pos, ' ')),
                    |pos| parse_type(s, pos, 'T', tm)),
                    |pos| parse_char(s, pos, ' ')),
                    |pos| parse_type(s, pos, 'Y', tm))
          }
          'D' | 'x' => {
                // FIXME(#3724): cleanup
                result::chain(
                result::chain(
                result::chain(
                result::chain(
                    move parse_type(s, pos, 'm', tm),
                    |pos| parse_char(s, pos, '/')),
                    |pos| parse_type(s, pos, 'd', tm)),
                    |pos| parse_char(s, pos, '/')),
                    |pos| parse_type(s, pos, 'y', tm))
          }
          'd' => match match_digits(s, pos, 2u, false) {
            Some(item) => { let (v, pos) = item; tm.tm_mday = v; Ok(pos) }
            None => Err(~"Invalid day of the month")
          },
          'e' => match match_digits(s, pos, 2u, true) {
            Some(item) => { let (v, pos) = item; tm.tm_mday = v; Ok(pos) }
            None => Err(~"Invalid day of the month")
          },
          'F' => {
                // FIXME(#3724): cleanup
                result::chain(
                result::chain(
                result::chain(
                result::chain(
                    move parse_type(s, pos, 'Y', tm),
                    |pos| parse_char(s, pos, '-')),
                    |pos| parse_type(s, pos, 'm', tm)),
                    |pos| parse_char(s, pos, '-')),
                    |pos| parse_type(s, pos, 'd', tm))
          }
          'H' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 2u, false) {
              Some(item) => { let (v, pos) = item; tm.tm_hour = v; Ok(pos) }
              None => Err(~"Invalid hour")
            }
          }
          'I' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 2u, false) {
              Some(item) => {
                  let (v, pos) = item;
                  tm.tm_hour = if v == 12_i32 { 0_i32 } else { v };
                  Ok(pos)
              }
              None => Err(~"Invalid hour")
            }
          }
          'j' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 3u, false) {
              Some(item) => {
                let (v, pos) = item;
                tm.tm_yday = v - 1_i32;
                Ok(pos)
              }
              None => Err(~"Invalid year")
            }
          }
          'k' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 2u, true) {
              Some(item) => { let (v, pos) = item; tm.tm_hour = v; Ok(pos) }
              None => Err(~"Invalid hour")
            }
          }
          'l' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 2u, true) {
              Some(item) => {
                  let (v, pos) = item;
                  tm.tm_hour = if v == 12_i32 { 0_i32 } else { v };
                  Ok(pos)
              }
              None => Err(~"Invalid hour")
            }
          }
          'M' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 2u, false) {
              Some(item) => { let (v, pos) = item; tm.tm_min = v; Ok(pos) }
              None => Err(~"Invalid minute")
            }
          }
          'm' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 2u, false) {
              Some(item) => {
                let (v, pos) = item;
                tm.tm_mon = v - 1_i32;
                Ok(pos)
              }
              None => Err(~"Invalid month")
            }
          }
          'n' => parse_char(s, pos, '\n'),
          'P' => match match_strs(s, pos,
                                  ~[(~"am", 0_i32), (~"pm", 12_i32)]) {

            Some(item) => { let (v, pos) = item; tm.tm_hour += v; Ok(pos) }
            None => Err(~"Invalid hour")
          },
          'p' => match match_strs(s, pos,
                                  ~[(~"AM", 0_i32), (~"PM", 12_i32)]) {

            Some(item) => { let (v, pos) = item; tm.tm_hour += v; Ok(pos) }
            None => Err(~"Invalid hour")
          },
          'R' => {
                // FIXME(#3724): cleanup
                result::chain(
                result::chain(
                    move parse_type(s, pos, 'H', tm),
                    |pos| parse_char(s, pos, ':')),
                    |pos| parse_type(s, pos, 'M', tm))
          }
          'r' => {
                // FIXME(#3724): cleanup
                result::chain(
                result::chain(
                result::chain(
                result::chain(
                result::chain(
                result::chain(
                    move parse_type(s, pos, 'I', tm),
                    |pos| parse_char(s, pos, ':')),
                    |pos| parse_type(s, pos, 'M', tm)),
                    |pos| parse_char(s, pos, ':')),
                    |pos| parse_type(s, pos, 'S', tm)),
                    |pos| parse_char(s, pos, ' ')),
                    |pos| parse_type(s, pos, 'p', tm))
          }
          'S' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 2u, false) {
              Some(item) => {
                let (v, pos) = item;
                tm.tm_sec = v;
                Ok(pos)
              }
              None => Err(~"Invalid second")
            }
          }
          //'s' {}
          'T' | 'X' => {
                // FIXME(#3724): cleanup
                result::chain(
                result::chain(
                result::chain(
                result::chain(
                    move parse_type(s, pos, 'H', tm),
                    |pos| parse_char(s, pos, ':')),
                    |pos| parse_type(s, pos, 'M', tm)),
                    |pos| parse_char(s, pos, ':')),
                    |pos| parse_type(s, pos, 'S', tm))
          }
          't' => parse_char(s, pos, '\t'),
          'u' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 1u, false) {
              Some(item) => {
                let (v, pos) = item;
                tm.tm_wday = v;
                Ok(pos)
              }
              None => Err(~"Invalid weekday")
            }
          }
          'v' => {
                // FIXME(#3724): cleanup
                result::chain(
                result::chain(
                result::chain(
                result::chain(
                    move parse_type(s, pos, 'e', tm),
                    |pos| parse_char(s, pos, '-')),
                    |pos| parse_type(s, pos, 'b', tm)),
                    |pos| parse_char(s, pos, '-')),
                    |pos| parse_type(s, pos, 'Y', tm))
          }
          //'W' {}
          'w' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 1u, false) {
              Some(item) => { let (v, pos) = item; tm.tm_wday = v; Ok(pos) }
              None => Err(~"Invalid weekday")
            }
          }
          //'X' {}
          //'x' {}
          'Y' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 4u, false) {
              Some(item) => {
                let (v, pos) = item;
                tm.tm_year = v - 1900_i32;
                Ok(pos)
              }
              None => Err(~"Invalid weekday")
            }
          }
          'y' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 2u, false) {
              Some(item) => {
                let (v, pos) = item;
                tm.tm_year = v - 1900_i32;
                Ok(pos)
              }
              None => Err(~"Invalid weekday")
            }
          }
          'Z' => {
            if match_str(s, pos, ~"UTC") || match_str(s, pos, ~"GMT") {
                tm.tm_gmtoff = 0_i32;
                tm.tm_zone = ~"UTC";
                Ok(pos + 3u)
            } else {
                // It's odd, but to maintain compatibility with c's
                // strptime we ignore the timezone.
                let mut pos = pos;
                let len = str::len(s);
                while pos < len {
                    let range = str::char_range_at(s, pos);
                    pos = range.next;
                    if range.ch == ' ' { break; }
                }

                Ok(pos)
            }
          }
          'z' => {
            let range = str::char_range_at(s, pos);

            if range.ch == '+' || range.ch == '-' {
                match match_digits(s, range.next, 4u, false) {
                  Some(item) => {
                    let (v, pos) = item;
                    if v == 0_i32 {
                        tm.tm_gmtoff = 0_i32;
                        tm.tm_zone = ~"UTC";
                    }

                    Ok(pos)
                  }
                  None => Err(~"Invalid zone offset")
                }
            } else {
                Err(~"Invalid zone offset")
            }
          }
          '%' => parse_char(s, pos, '%'),
          ch => {
            Err(fmt!("unknown formatting type: %?", str::from_char(ch)))
          }
        }
    }

    do io::with_str_reader(str::from_slice(format)) |rdr| {
        let mut tm = Tm {
            tm_sec: 0_i32,
            tm_min: 0_i32,
            tm_hour: 0_i32,
            tm_mday: 0_i32,
            tm_mon: 0_i32,
            tm_year: 0_i32,
            tm_wday: 0_i32,
            tm_yday: 0_i32,
            tm_isdst: 0_i32,
            tm_gmtoff: 0_i32,
            tm_zone: ~"",
            tm_nsec: 0_i32,
        };
        let mut pos = 0u;
        let len = str::len(s);
        let mut result = Err(~"Invalid time");

        while !rdr.eof() && pos < len {
            let range = str::char_range_at(s, pos);
            let ch = range.ch;
            let next = range.next;

            match rdr.read_char() {
                '%' => {
                    match parse_type(s, pos, rdr.read_char(), &mut tm) {
                        Ok(next) => pos = next,
                        Err(move e) => { result = Err(move e); break; }
                    }
                },
                c => {
                    if c != ch { break }
                    pos = next;
                }
            }
        }

        if pos == len && rdr.eof() {
            Ok(Tm {
                tm_sec: tm.tm_sec,
                tm_min: tm.tm_min,
                tm_hour: tm.tm_hour,
                tm_mday: tm.tm_mday,
                tm_mon: tm.tm_mon,
                tm_year: tm.tm_year,
                tm_wday: tm.tm_wday,
                tm_yday: tm.tm_yday,
                tm_isdst: tm.tm_isdst,
                tm_gmtoff: tm.tm_gmtoff,
                tm_zone: copy tm.tm_zone,
                tm_nsec: tm.tm_nsec,
            })
        } else { move result }
    }
}

priv fn do_strftime(format: &str, tm: &Tm) -> ~str {
    fn parse_type(ch: char, tm: &Tm) -> ~str {
        //FIXME (#2350): Implement missing types.
      let die = || fmt!("strftime: can't understand this format %c ", ch);
        match ch {
          'A' => match tm.tm_wday as int {
            0 => ~"Sunday",
            1 => ~"Monday",
            2 => ~"Tuesday",
            3 => ~"Wednesday",
            4 => ~"Thursday",
            5 => ~"Friday",
            6 => ~"Saturday",
            _ => die()
          },
         'a' => match tm.tm_wday as int {
            0 => ~"Sun",
            1 => ~"Mon",
            2 => ~"Tue",
            3 => ~"Wed",
            4 => ~"Thu",
            5 => ~"Fri",
            6 => ~"Sat",
            _ => die()
          },
          'B' => match tm.tm_mon as int {
            0 => ~"January",
            1 => ~"February",
            2 => ~"March",
            3 => ~"April",
            4 => ~"May",
            5 => ~"June",
            6 => ~"July",
            7 => ~"August",
            8 => ~"September",
            9 => ~"October",
            10 => ~"November",
            11 => ~"December",
            _ => die()
          },
          'b' | 'h' => match tm.tm_mon as int {
            0 => ~"Jan",
            1 => ~"Feb",
            2 => ~"Mar",
            3 => ~"Apr",
            4 => ~"May",
            5 => ~"Jun",
            6 => ~"Jul",
            7 => ~"Aug",
            8 => ~"Sep",
            9 => ~"Oct",
            10 => ~"Nov",
            11 => ~"Dec",
            _  => die()
          },
          'C' => fmt!("%02d", (tm.tm_year as int + 1900) / 100),
          'c' => {
            fmt!("%s %s %s %s %s",
                parse_type('a', tm),
                parse_type('b', tm),
                parse_type('e', tm),
                parse_type('T', tm),
                parse_type('Y', tm))
          }
          'D' | 'x' => {
            fmt!("%s/%s/%s",
                parse_type('m', tm),
                parse_type('d', tm),
                parse_type('y', tm))
          }
          'd' => fmt!("%02d", tm.tm_mday as int),
          'e' => fmt!("%2d", tm.tm_mday as int),
          'F' => {
            fmt!("%s-%s-%s",
                parse_type('Y', tm),
                parse_type('m', tm),
                parse_type('d', tm))
          }
          //'G' {}
          //'g' {}
          'H' => fmt!("%02d", tm.tm_hour as int),
          'I' => {
            let mut h = tm.tm_hour as int;
            if h == 0 { h = 12 }
            if h > 12 { h -= 12 }
            fmt!("%02d", h)
          }
          'j' => fmt!("%03d", tm.tm_yday as int + 1),
          'k' => fmt!("%2d", tm.tm_hour as int),
          'l' => {
            let mut h = tm.tm_hour as int;
            if h == 0 { h = 12 }
            if h > 12 { h -= 12 }
            fmt!("%2d", h)
          }
          'M' => fmt!("%02d", tm.tm_min as int),
          'm' => fmt!("%02d", tm.tm_mon as int + 1),
          'n' => ~"\n",
          'P' => if tm.tm_hour as int < 12 { ~"am" } else { ~"pm" },
          'p' => if tm.tm_hour as int < 12 { ~"AM" } else { ~"PM" },
          'R' => {
            fmt!("%s:%s",
                parse_type('H', tm),
                parse_type('M', tm))
          }
          'r' => {
            fmt!("%s:%s:%s %s",
                parse_type('I', tm),
                parse_type('M', tm),
                parse_type('S', tm),
                parse_type('p', tm))
          }
          'S' => fmt!("%02d", tm.tm_sec as int),
          's' => fmt!("%d", tm.to_timespec().sec as int),
          'T' | 'X' => {
            fmt!("%s:%s:%s",
                parse_type('H', tm),
                parse_type('M', tm),
                parse_type('S', tm))
          }
          't' => ~"\t",
          //'U' {}
          'u' => {
            let i = tm.tm_wday as int;
            int::str(if i == 0 { 7 } else { i })
          }
          //'V' {}
          'v' => {
            fmt!("%s-%s-%s",
                parse_type('e', tm),
                parse_type('b', tm),
                parse_type('Y', tm))
          }
          //'W' {}
          'w' => int::str(tm.tm_wday as int),
          //'X' {}
          //'x' {}
          'Y' => int::str(tm.tm_year as int + 1900),
          'y' => fmt!("%02d", (tm.tm_year as int + 1900) % 100),
          'Z' => copy tm.tm_zone,
          'z' => {
            let sign = if tm.tm_gmtoff > 0_i32 { '+' } else { '-' };
            let mut m = i32::abs(tm.tm_gmtoff) / 60_i32;
            let h = m / 60_i32;
            m -= h * 60_i32;
            fmt!("%c%02d%02d", sign, h as int, m as int)
          }
          //'+' {}
          '%' => ~"%",
          _   => die()
        }
    }

    let mut buf = ~"";

    do io::with_str_reader(str::from_slice(format)) |rdr| {
        while !rdr.eof() {
            match rdr.read_char() {
                '%' => buf += parse_type(rdr.read_char(), tm),
                ch => str::push_char(&mut buf, ch)
            }
        }
    }

    move buf
}

#[cfg(test)]
mod tests {
    #[legacy_exports];

    #[test]
    fn test_get_time() {
        const some_recent_date: i64 = 1325376000i64; // 2012-01-01T00:00:00Z
        const some_future_date: i64 = 1577836800i64; // 2020-01-01T00:00:00Z

        let tv1 = get_time();
        log(debug, ~"tv1=" + uint::str(tv1.sec as uint) + ~" sec + "
                   + uint::str(tv1.nsec as uint) + ~" nsec");

        assert tv1.sec > some_recent_date;
        assert tv1.nsec < 1000000000i32;

        let tv2 = get_time();
        log(debug, ~"tv2=" + uint::str(tv2.sec as uint) + ~" sec + "
                   + uint::str(tv2.nsec as uint) + ~" nsec");

        assert tv2.sec >= tv1.sec;
        assert tv2.sec < some_future_date;
        assert tv2.nsec < 1000000000i32;
        if tv2.sec == tv1.sec {
            assert tv2.nsec >= tv1.nsec;
        }
    }

    #[test]
    fn test_precise_time() {
        let s0 = precise_time_s();
        let ns1 = precise_time_ns();

        log(debug, ~"s0=" + float::to_str(s0, 9u) + ~" sec");
        assert s0 > 0.;
        let ns0 = (s0 * 1000000000.) as u64;
        log(debug, ~"ns0=" + u64::str(ns0) + ~" ns");

        log(debug, ~"ns1=" + u64::str(ns1) + ~" ns");
        assert ns1 >= ns0;

        let ns2 = precise_time_ns();
        log(debug, ~"ns2=" + u64::str(ns2) + ~" ns");
        assert ns2 >= ns1;
    }

    #[test]
    fn test_at_utc() {
        os::setenv(~"TZ", ~"America/Los_Angeles");
        tzset();

        let time = Timespec::new(1234567890, 54321);
        let utc = at_utc(time);

        assert utc.tm_sec == 30_i32;
        assert utc.tm_min == 31_i32;
        assert utc.tm_hour == 23_i32;
        assert utc.tm_mday == 13_i32;
        assert utc.tm_mon == 1_i32;
        assert utc.tm_year == 109_i32;
        assert utc.tm_wday == 5_i32;
        assert utc.tm_yday == 43_i32;
        assert utc.tm_isdst == 0_i32;
        assert utc.tm_gmtoff == 0_i32;
        assert utc.tm_zone == ~"UTC";
        assert utc.tm_nsec == 54321_i32;
    }

    #[test]
    fn test_at() {
        os::setenv(~"TZ", ~"America/Los_Angeles");
        tzset();

        let time = Timespec::new(1234567890, 54321);
        let local = at(time);

        error!("time_at: %?", local);

        assert local.tm_sec == 30_i32;
        assert local.tm_min == 31_i32;
        assert local.tm_hour == 15_i32;
        assert local.tm_mday == 13_i32;
        assert local.tm_mon == 1_i32;
        assert local.tm_year == 109_i32;
        assert local.tm_wday == 5_i32;
        assert local.tm_yday == 43_i32;
        assert local.tm_isdst == 0_i32;
        assert local.tm_gmtoff == -28800_i32;

        // FIXME (#2350): We should probably standardize on the timezone
        // abbreviation.
        let zone = &local.tm_zone;
        assert *zone == ~"PST" || *zone == ~"Pacific Standard Time";

        assert local.tm_nsec == 54321_i32;
    }

    #[test]
    fn test_to_timespec() {
        os::setenv(~"TZ", ~"America/Los_Angeles");
        tzset();

        let time = Timespec::new(1234567890, 54321);
        let utc = at_utc(time);

        assert utc.to_timespec() == time;
        assert utc.to_local().to_timespec() == time;
    }

    #[test]
    fn test_conversions() {
        os::setenv(~"TZ", ~"America/Los_Angeles");
        tzset();

        let time = Timespec::new(1234567890, 54321);
        let utc = at_utc(time);
        let local = at(time);

        assert local.to_local() == local;
        assert local.to_utc() == utc;
        assert local.to_utc().to_local() == local;
        assert utc.to_utc() == utc;
        assert utc.to_local() == local;
        assert utc.to_local().to_utc() == utc;
    }

    #[test]
    fn test_strptime() {
        os::setenv(~"TZ", ~"America/Los_Angeles");
        tzset();

        match strptime(~"", ~"") {
          Ok(ref tm) => {
            assert tm.tm_sec == 0_i32;
            assert tm.tm_min == 0_i32;
            assert tm.tm_hour == 0_i32;
            assert tm.tm_mday == 0_i32;
            assert tm.tm_mon == 0_i32;
            assert tm.tm_year == 0_i32;
            assert tm.tm_wday == 0_i32;
            assert tm.tm_isdst== 0_i32;
            assert tm.tm_gmtoff == 0_i32;
            assert tm.tm_zone == ~"";
            assert tm.tm_nsec == 0_i32;
          }
          Err(_) => ()
        }

        let format = ~"%a %b %e %T %Y";
        assert strptime(~"", format) == Err(~"Invalid time");
        assert strptime(~"Fri Feb 13 15:31:30", format)
            == Err(~"Invalid time");

        match strptime(~"Fri Feb 13 15:31:30 2009", format) {
          Err(copy e) => fail e,
          Ok(ref tm) => {
            assert tm.tm_sec == 30_i32;
            assert tm.tm_min == 31_i32;
            assert tm.tm_hour == 15_i32;
            assert tm.tm_mday == 13_i32;
            assert tm.tm_mon == 1_i32;
            assert tm.tm_year == 109_i32;
            assert tm.tm_wday == 5_i32;
            assert tm.tm_yday == 0_i32;
            assert tm.tm_isdst == 0_i32;
            assert tm.tm_gmtoff == 0_i32;
            assert tm.tm_zone == ~"";
            assert tm.tm_nsec == 0_i32;
          }
        }

        fn test(s: &str, format: &str) -> bool {
            match strptime(s, format) {
              Ok(ref tm) => tm.strftime(format) == str::from_slice(s),
              Err(copy e) => fail e
            }
        }

        for vec::each([
            ~"Sunday",
            ~"Monday",
            ~"Tuesday",
            ~"Wednesday",
            ~"Thursday",
            ~"Friday",
            ~"Saturday"
        ]) |day| {
            assert test(*day, ~"%A");
        }

        for vec::each([
            ~"Sun",
            ~"Mon",
            ~"Tue",
            ~"Wed",
            ~"Thu",
            ~"Fri",
            ~"Sat"
        ]) |day| {
            assert test(*day, ~"%a");
        }

        for vec::each([
            ~"January",
            ~"February",
            ~"March",
            ~"April",
            ~"May",
            ~"June",
            ~"July",
            ~"August",
            ~"September",
            ~"October",
            ~"November",
            ~"December"
        ]) |day| {
            assert test(*day, ~"%B");
        }

        for vec::each([
            ~"Jan",
            ~"Feb",
            ~"Mar",
            ~"Apr",
            ~"May",
            ~"Jun",
            ~"Jul",
            ~"Aug",
            ~"Sep",
            ~"Oct",
            ~"Nov",
            ~"Dec"
        ]) |day| {
            assert test(*day, ~"%b");
        }

        assert test(~"19", ~"%C");
        assert test(~"Fri Feb 13 23:31:30 2009", ~"%c");
        assert test(~"02/13/09", ~"%D");
        assert test(~"03", ~"%d");
        assert test(~"13", ~"%d");
        assert test(~" 3", ~"%e");
        assert test(~"13", ~"%e");
        assert test(~"2009-02-13", ~"%F");
        assert test(~"03", ~"%H");
        assert test(~"13", ~"%H");
        assert test(~"03", ~"%I"); // FIXME (#2350): flesh out
        assert test(~"11", ~"%I"); // FIXME (#2350): flesh out
        assert test(~"044", ~"%j");
        assert test(~" 3", ~"%k");
        assert test(~"13", ~"%k");
        assert test(~" 1", ~"%l");
        assert test(~"11", ~"%l");
        assert test(~"03", ~"%M");
        assert test(~"13", ~"%M");
        assert test(~"\n", ~"%n");
        assert test(~"am", ~"%P");
        assert test(~"pm", ~"%P");
        assert test(~"AM", ~"%p");
        assert test(~"PM", ~"%p");
        assert test(~"23:31", ~"%R");
        assert test(~"11:31:30 AM", ~"%r");
        assert test(~"11:31:30 PM", ~"%r");
        assert test(~"03", ~"%S");
        assert test(~"13", ~"%S");
        assert test(~"15:31:30", ~"%T");
        assert test(~"\t", ~"%t");
        assert test(~"1", ~"%u");
        assert test(~"7", ~"%u");
        assert test(~"13-Feb-2009", ~"%v");
        assert test(~"0", ~"%w");
        assert test(~"6", ~"%w");
        assert test(~"2009", ~"%Y");
        assert test(~"09", ~"%y");
        assert result::unwrap(strptime(~"UTC", ~"%Z")).tm_zone == ~"UTC";
        assert result::unwrap(strptime(~"PST", ~"%Z")).tm_zone == ~"";
        assert result::unwrap(strptime(~"-0000", ~"%z")).tm_gmtoff == 0;
        assert result::unwrap(strptime(~"-0800", ~"%z")).tm_gmtoff == 0;
        assert test(~"%", ~"%%");
    }

    #[test]
    fn test_ctime() {
        os::setenv(~"TZ", ~"America/Los_Angeles");
        tzset();

        let time = Timespec::new(1234567890, 54321);
        let utc   = at_utc(time);
        let local = at(time);

        error!("test_ctime: %? %?", utc.ctime(), local.ctime());

        assert utc.ctime()   == ~"Fri Feb 13 23:31:30 2009";
        assert local.ctime() == ~"Fri Feb 13 15:31:30 2009";
    }

    #[test]
    fn test_strftime() {
        os::setenv(~"TZ", ~"America/Los_Angeles");
        tzset();

        let time = Timespec::new(1234567890, 54321);
        let utc = at_utc(time);
        let local = at(time);

        assert local.strftime(~"") == ~"";
        assert local.strftime(~"%A") == ~"Friday";
        assert local.strftime(~"%a") == ~"Fri";
        assert local.strftime(~"%B") == ~"February";
        assert local.strftime(~"%b") == ~"Feb";
        assert local.strftime(~"%C") == ~"20";
        assert local.strftime(~"%c") == ~"Fri Feb 13 15:31:30 2009";
        assert local.strftime(~"%D") == ~"02/13/09";
        assert local.strftime(~"%d") == ~"13";
        assert local.strftime(~"%e") == ~"13";
        assert local.strftime(~"%F") == ~"2009-02-13";
        // assert local.strftime("%G") == "2009";
        // assert local.strftime("%g") == "09";
        assert local.strftime(~"%H") == ~"15";
        assert local.strftime(~"%I") == ~"03";
        assert local.strftime(~"%j") == ~"044";
        assert local.strftime(~"%k") == ~"15";
        assert local.strftime(~"%l") == ~" 3";
        assert local.strftime(~"%M") == ~"31";
        assert local.strftime(~"%m") == ~"02";
        assert local.strftime(~"%n") == ~"\n";
        assert local.strftime(~"%P") == ~"pm";
        assert local.strftime(~"%p") == ~"PM";
        assert local.strftime(~"%R") == ~"15:31";
        assert local.strftime(~"%r") == ~"03:31:30 PM";
        assert local.strftime(~"%S") == ~"30";
        assert local.strftime(~"%s") == ~"1234567890";
        assert local.strftime(~"%T") == ~"15:31:30";
        assert local.strftime(~"%t") == ~"\t";
        // assert local.strftime("%U") == "06";
        assert local.strftime(~"%u") == ~"5";
        // assert local.strftime("%V") == "07";
        assert local.strftime(~"%v") == ~"13-Feb-2009";
        // assert local.strftime("%W") == "06";
        assert local.strftime(~"%w") == ~"5";
        // handle "%X"
        // handle "%x"
        assert local.strftime(~"%Y") == ~"2009";
        assert local.strftime(~"%y") == ~"09";

        // FIXME (#2350): We should probably standardize on the timezone
        // abbreviation.
        let zone = local.strftime(~"%Z");
        assert zone == ~"PST" || zone == ~"Pacific Standard Time";

        assert local.strftime(~"%z") == ~"-0800";
        assert local.strftime(~"%%") == ~"%";

        // FIXME (#2350): We should probably standardize on the timezone
        // abbreviation.
        let rfc822 = local.rfc822();
        let prefix = ~"Fri, 13 Feb 2009 15:31:30 ";
        assert rfc822 == prefix + ~"PST" ||
               rfc822 == prefix + ~"Pacific Standard Time";

        assert local.ctime() == ~"Fri Feb 13 15:31:30 2009";
        assert local.rfc822z() == ~"Fri, 13 Feb 2009 15:31:30 -0800";
        assert local.rfc3339() == ~"2009-02-13T15:31:30-08:00";

        assert utc.ctime() == ~"Fri Feb 13 23:31:30 2009";
        assert utc.rfc822() == ~"Fri, 13 Feb 2009 23:31:30 GMT";
        assert utc.rfc822z() == ~"Fri, 13 Feb 2009 23:31:30 -0000";
        assert utc.rfc3339() == ~"2009-02-13T23:31:30Z";
    }
}
