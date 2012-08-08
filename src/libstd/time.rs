import libc::{c_char, c_int, c_long, size_t, time_t};
import io::reader;
import result::{result, ok, err};

export
    timespec,
    get_time,
    precise_time_ns,
    precise_time_s,
    tzset,
    tm,
    empty_tm,
    now,
    at,
    now_utc,
    at_utc,
    strptime;

#[abi = "cdecl"]
extern mod rustrt {
    fn get_time(&sec: i64, &nsec: i32);
    fn precise_time_ns(&ns: u64);

    fn rust_tzset();
    // FIXME: The i64 values can be passed by-val when #2064 is fixed.
    fn rust_gmtime(&&sec: i64, &&nsec: i32, &&result: tm);
    fn rust_localtime(&&sec: i64, &&nsec: i32, &&result: tm);
    fn rust_timegm(&&tm: tm, &sec: i64);
    fn rust_mktime(&&tm: tm, &sec: i64);
}

/// A record specifying a time value in seconds and microseconds.
type timespec = {sec: i64, nsec: i32};

/**
 * Returns the current time as a `timespec` containing the seconds and
 * microseconds since 1970-01-01T00:00:00Z.
 */
fn get_time() -> timespec {
    let mut sec = 0i64;
    let mut nsec = 0i32;
    rustrt::get_time(sec, nsec);
    return {sec: sec, nsec: nsec};
}

/**
 * Returns the current value of a high-resolution performance counter
 * in nanoseconds since an unspecified epoch.
 */
fn precise_time_ns() -> u64 {
    let mut ns = 0u64;
    rustrt::precise_time_ns(ns);
    ns
}

/**
 * Returns the current value of a high-resolution performance counter
 * in seconds since an unspecified epoch.
 */
fn precise_time_s() -> float {
    return (precise_time_ns() as float) / 1000000000.;
}

fn tzset() {
    rustrt::rust_tzset();
}

type tm_ = {
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
};

enum tm {
    tm_(tm_)
}

fn empty_tm() -> tm {
    tm_({
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
    })
}

/// Returns the specified time in UTC
fn at_utc(clock: timespec) -> tm {
    let mut {sec, nsec} = clock;
    let mut tm = empty_tm();
    rustrt::rust_gmtime(sec, nsec, tm);
    tm
}

/// Returns the current time in UTC
fn now_utc() -> tm {
    at_utc(get_time())
}

/// Returns the specified time in the local timezone
fn at(clock: timespec) -> tm {
    let mut {sec, nsec} = clock;
    let mut tm = empty_tm();
    rustrt::rust_localtime(sec, nsec, tm);
    tm
}

/// Returns the current time in the local timezone
fn now() -> tm {
    at(get_time())
}

/// Parses the time from the string according to the format string.
fn strptime(s: ~str, format: ~str) -> result<tm, ~str> {
    type tm_mut = {
       mut tm_sec: i32,
       mut tm_min: i32,
       mut tm_hour: i32,
       mut tm_mday: i32,
       mut tm_mon: i32,
       mut tm_year: i32,
       mut tm_wday: i32,
       mut tm_yday: i32,
       mut tm_isdst: i32,
       mut tm_gmtoff: i32,
       mut tm_zone: ~str,
       mut tm_nsec: i32,
    };

    fn match_str(s: ~str, pos: uint, needle: ~str) -> bool {
        let mut i = pos;
        for str::each(needle) |ch| {
            if s[i] != ch {
                return false;
            }
            i += 1u;
        }
        return true;
    }

    fn match_strs(s: ~str, pos: uint, strs: ~[(~str, i32)])
      -> option<(i32, uint)> {
        let mut i = 0u;
        let len = vec::len(strs);
        while i < len {
            let (needle, value) = strs[i];

            if match_str(s, pos, needle) {
                return some((value, pos + str::len(needle)));
            }
            i += 1u;
        }

        none
    }

    fn match_digits(s: ~str, pos: uint, digits: uint, ws: bool)
      -> option<(i32, uint)> {
        let mut pos = pos;
        let mut value = 0_i32;

        let mut i = 0u;
        while i < digits {
            let {ch, next} = str::char_range_at(s, pos);
            pos = next;

            match ch {
              '0' to '9' => {
                value = value * 10_i32 + (ch as i32 - '0' as i32);
              }
              ' ' if ws => (),
              _ => return none
            }
            i += 1u;
        }

        some((value, pos))
    }

    fn parse_char(s: ~str, pos: uint, c: char) -> result<uint, ~str> {
        let {ch, next} = str::char_range_at(s, pos);

        if c == ch {
            ok(next)
        } else {
            err(fmt!{"Expected %?, found %?",
                str::from_char(c),
                str::from_char(ch)})
        }
    }

    fn parse_type(s: ~str, pos: uint, ch: char, tm: tm_mut)
      -> result<uint, ~str> {
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
            some(item) => { let (v, pos) = item; tm.tm_wday = v; ok(pos) }
            none => err(~"Invalid day")
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
            some(item) => { let (v, pos) = item; tm.tm_wday = v; ok(pos) }
            none => err(~"Invalid day")
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
            some(item) => { let (v, pos) = item; tm.tm_mon = v; ok(pos) }
            none => err(~"Invalid month")
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
            some(item) => { let (v, pos) = item; tm.tm_mon = v; ok(pos) }
            none => err(~"Invalid month")
          },
          'C' => match match_digits(s, pos, 2u, false) {
            some(item) => {
                let (v, pos) = item;
                  tm.tm_year += (v * 100_i32) - 1900_i32;
                  ok(pos)
              }
            none => err(~"Invalid year")
          },
          'c' => {
            parse_type(s, pos, 'a', tm)
                .chain(|pos| parse_char(s, pos, ' '))
                .chain(|pos| parse_type(s, pos, 'b', tm))
                .chain(|pos| parse_char(s, pos, ' '))
                .chain(|pos| parse_type(s, pos, 'e', tm))
                .chain(|pos| parse_char(s, pos, ' '))
                .chain(|pos| parse_type(s, pos, 'T', tm))
                .chain(|pos| parse_char(s, pos, ' '))
                .chain(|pos| parse_type(s, pos, 'Y', tm))
          }
          'D' | 'x' => {
            parse_type(s, pos, 'm', tm)
                .chain(|pos| parse_char(s, pos, '/'))
                .chain(|pos| parse_type(s, pos, 'd', tm))
                .chain(|pos| parse_char(s, pos, '/'))
                .chain(|pos| parse_type(s, pos, 'y', tm))
          }
          'd' => match match_digits(s, pos, 2u, false) {
            some(item) => { let (v, pos) = item; tm.tm_mday = v; ok(pos) }
            none => err(~"Invalid day of the month")
          },
          'e' => match match_digits(s, pos, 2u, true) {
            some(item) => { let (v, pos) = item; tm.tm_mday = v; ok(pos) }
            none => err(~"Invalid day of the month")
          },
          'F' => {
            parse_type(s, pos, 'Y', tm)
                .chain(|pos| parse_char(s, pos, '-'))
                .chain(|pos| parse_type(s, pos, 'm', tm))
                .chain(|pos| parse_char(s, pos, '-'))
                .chain(|pos| parse_type(s, pos, 'd', tm))
          }
          'H' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 2u, false) {
              some(item) => { let (v, pos) = item; tm.tm_hour = v; ok(pos) }
              none => err(~"Invalid hour")
            }
          }
          'I' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 2u, false) {
              some(item) => {
                  let (v, pos) = item;
                  tm.tm_hour = if v == 12_i32 { 0_i32 } else { v };
                  ok(pos)
              }
              none => err(~"Invalid hour")
            }
          }
          'j' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 3u, false) {
              some(item) => {
                let (v, pos) = item;
                tm.tm_yday = v - 1_i32;
                ok(pos)
              }
              none => err(~"Invalid year")
            }
          }
          'k' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 2u, true) {
              some(item) => { let (v, pos) = item; tm.tm_hour = v; ok(pos) }
              none => err(~"Invalid hour")
            }
          }
          'l' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 2u, true) {
              some(item) => {
                  let (v, pos) = item;
                  tm.tm_hour = if v == 12_i32 { 0_i32 } else { v };
                  ok(pos)
              }
              none => err(~"Invalid hour")
            }
          }
          'M' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 2u, false) {
              some(item) => { let (v, pos) = item; tm.tm_min = v; ok(pos) }
              none => err(~"Invalid minute")
            }
          }
          'm' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 2u, false) {
              some(item) => {
                let (v, pos) = item;
                tm.tm_mon = v - 1_i32;
                ok(pos)
              }
              none => err(~"Invalid month")
            }
          }
          'n' => parse_char(s, pos, '\n'),
          'P' => match match_strs(s, pos,
                                  ~[(~"am", 0_i32), (~"pm", 12_i32)]) {

            some(item) => { let (v, pos) = item; tm.tm_hour += v; ok(pos) }
            none => err(~"Invalid hour")
          },
          'p' => match match_strs(s, pos,
                                  ~[(~"AM", 0_i32), (~"PM", 12_i32)]) {

            some(item) => { let (v, pos) = item; tm.tm_hour += v; ok(pos) }
            none => err(~"Invalid hour")
          },
          'R' => {
            parse_type(s, pos, 'H', tm)
                .chain(|pos| parse_char(s, pos, ':'))
                .chain(|pos| parse_type(s, pos, 'M', tm))
          }
          'r' => {
            parse_type(s, pos, 'I', tm)
                .chain(|pos| parse_char(s, pos, ':'))
                .chain(|pos| parse_type(s, pos, 'M', tm))
                .chain(|pos| parse_char(s, pos, ':'))
                .chain(|pos| parse_type(s, pos, 'S', tm))
                .chain(|pos| parse_char(s, pos, ' '))
                .chain(|pos| parse_type(s, pos, 'p', tm))
          }
          'S' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 2u, false) {
              some(item) => {
                let (v, pos) = item;
                tm.tm_sec = v;
                ok(pos)
              }
              none => err(~"Invalid second")
            }
          }
          //'s' {}
          'T' | 'X' => {
            parse_type(s, pos, 'H', tm)
                .chain(|pos| parse_char(s, pos, ':'))
                .chain(|pos| parse_type(s, pos, 'M', tm))
                .chain(|pos| parse_char(s, pos, ':'))
                .chain(|pos| parse_type(s, pos, 'S', tm))
          }
          't' => parse_char(s, pos, '\t'),
          'u' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 1u, false) {
              some(item) => {
                let (v, pos) = item;
                tm.tm_wday = v;
                ok(pos)
              }
              none => err(~"Invalid weekday")
            }
          }
          'v' => {
            parse_type(s, pos, 'e', tm)
                .chain(|pos| parse_char(s, pos, '-'))
                .chain(|pos| parse_type(s, pos, 'b', tm))
                .chain(|pos| parse_char(s, pos, '-'))
                .chain(|pos| parse_type(s, pos, 'Y', tm))
          }
          //'W' {}
          'w' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 1u, false) {
              some(item) => { let (v, pos) = item; tm.tm_wday = v; ok(pos) }
              none => err(~"Invalid weekday")
            }
          }
          //'X' {}
          //'x' {}
          'Y' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 4u, false) {
              some(item) => {
                let (v, pos) = item;
                tm.tm_year = v - 1900_i32;
                ok(pos)
              }
              none => err(~"Invalid weekday")
            }
          }
          'y' => {
            // FIXME (#2350): range check.
            match match_digits(s, pos, 2u, false) {
              some(item) => {
                let (v, pos) = item;
                tm.tm_year = v - 1900_i32;
                ok(pos)
              }
              none => err(~"Invalid weekday")
            }
          }
          'Z' => {
            if match_str(s, pos, ~"UTC") || match_str(s, pos, ~"GMT") {
                tm.tm_gmtoff = 0_i32;
                tm.tm_zone = ~"UTC";
                ok(pos + 3u)
            } else {
                // It's odd, but to maintain compatibility with c's
                // strptime we ignore the timezone.
                let mut pos = pos;
                let len = str::len(s);
                while pos < len {
                    let {ch, next} = str::char_range_at(s, pos);
                    pos = next;
                    if ch == ' ' { break; }
                }

                ok(pos)
            }
          }
          'z' => {
            let {ch, next} = str::char_range_at(s, pos);

            if ch == '+' || ch == '-' {
                match match_digits(s, next, 4u, false) {
                  some(item) => {
                    let (v, pos) = item;
                    if v == 0_i32 {
                        tm.tm_gmtoff = 0_i32;
                        tm.tm_zone = ~"UTC";
                    }

                    ok(pos)
                  }
                  none => err(~"Invalid zone offset")
                }
            } else {
                err(~"Invalid zone offset")
            }
          }
          '%' => parse_char(s, pos, '%'),
          ch => {
            err(fmt!{"unknown formatting type: %?", str::from_char(ch)})
          }
        }
    }

    do io::with_str_reader(format) |rdr| {
        let tm = {
            mut tm_sec: 0_i32,
            mut tm_min: 0_i32,
            mut tm_hour: 0_i32,
            mut tm_mday: 0_i32,
            mut tm_mon: 0_i32,
            mut tm_year: 0_i32,
            mut tm_wday: 0_i32,
            mut tm_yday: 0_i32,
            mut tm_isdst: 0_i32,
            mut tm_gmtoff: 0_i32,
            mut tm_zone: ~"",
            mut tm_nsec: 0_i32,
        };
        let mut pos = 0u;
        let len = str::len(s);
        let mut result = err(~"Invalid time");

        while !rdr.eof() && pos < len {
            let {ch, next} = str::char_range_at(s, pos);

            match rdr.read_char() {
              '%' => match parse_type(s, pos, rdr.read_char(), tm) {
                ok(next) => pos = next,
                  err(e) => { result = err(e); break; }
              },
              c => {
                if c != ch { break }
                pos = next;
              }
            }
        }

        if pos == len && rdr.eof() {
            ok(tm_({
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
                tm_zone: tm.tm_zone,
                tm_nsec: tm.tm_nsec,
            }))
        } else { result }
    }
}

fn strftime(format: ~str, tm: tm) -> ~str {
    fn parse_type(ch: char, tm: tm) -> ~str {
        //FIXME (#2350): Implement missing types.
        match check ch {
          'A' => match check tm.tm_wday as int {
            0 => ~"Sunday",
            1 => ~"Monday",
            2 => ~"Tuesday",
            3 => ~"Wednesday",
            4 => ~"Thursday",
            5 => ~"Friday",
            6 => ~"Saturday"
          },
          'a' => match check tm.tm_wday as int {
            0 => ~"Sun",
            1 => ~"Mon",
            2 => ~"Tue",
            3 => ~"Wed",
            4 => ~"Thu",
            5 => ~"Fri",
            6 => ~"Sat"
          },
          'B' => match check tm.tm_mon as int {
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
            11 => ~"December"
          },
          'b' | 'h' => match check tm.tm_mon as int {
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
          },
          'C' => fmt!{"%02d", (tm.tm_year as int + 1900) / 100},
          'c' => {
            fmt!{"%s %s %s %s %s",
                parse_type('a', tm),
                parse_type('b', tm),
                parse_type('e', tm),
                parse_type('T', tm),
                parse_type('Y', tm)}
          }
          'D' | 'x' => {
            fmt!{"%s/%s/%s",
                parse_type('m', tm),
                parse_type('d', tm),
                parse_type('y', tm)}
          }
          'd' => fmt!{"%02d", tm.tm_mday as int},
          'e' => fmt!{"%2d", tm.tm_mday as int},
          'F' => {
            fmt!{"%s-%s-%s",
                parse_type('Y', tm),
                parse_type('m', tm),
                parse_type('d', tm)}
          }
          //'G' {}
          //'g' {}
          'H' => fmt!{"%02d", tm.tm_hour as int},
          'I' => {
            let mut h = tm.tm_hour as int;
            if h == 0 { h = 12 }
            if h > 12 { h -= 12 }
            fmt!{"%02d", h}
          }
          'j' => fmt!{"%03d", tm.tm_yday as int + 1},
          'k' => fmt!{"%2d", tm.tm_hour as int},
          'l' => {
            let mut h = tm.tm_hour as int;
            if h == 0 { h = 12 }
            if h > 12 { h -= 12 }
            fmt!{"%2d", h}
          }
          'M' => fmt!{"%02d", tm.tm_min as int},
          'm' => fmt!{"%02d", tm.tm_mon as int + 1},
          'n' => ~"\n",
          'P' => if tm.tm_hour as int < 12 { ~"am" } else { ~"pm" },
          'p' => if tm.tm_hour as int < 12 { ~"AM" } else { ~"PM" },
          'R' => {
            fmt!{"%s:%s",
                parse_type('H', tm),
                parse_type('M', tm)}
          }
          'r' => {
            fmt!{"%s:%s:%s %s",
                parse_type('I', tm),
                parse_type('M', tm),
                parse_type('S', tm),
                parse_type('p', tm)}
          }
          'S' => fmt!{"%02d", tm.tm_sec as int},
          's' => fmt!{"%d", tm.to_timespec().sec as int},
          'T' | 'X' => {
            fmt!{"%s:%s:%s",
                parse_type('H', tm),
                parse_type('M', tm),
                parse_type('S', tm)}
          }
          't' => ~"\t",
          //'U' {}
          'u' => {
            let i = tm.tm_wday as int;
            int::str(if i == 0 { 7 } else { i })
          }
          //'V' {}
          'v' => {
            fmt!{"%s-%s-%s",
                parse_type('e', tm),
                parse_type('b', tm),
                parse_type('Y', tm)}
          }
          //'W' {}
          'w' => int::str(tm.tm_wday as int),
          //'X' {}
          //'x' {}
          'Y' => int::str(tm.tm_year as int + 1900),
          'y' => fmt!{"%02d", (tm.tm_year as int + 1900) % 100},
          'Z' => tm.tm_zone,
          'z' => {
            let sign = if tm.tm_gmtoff > 0_i32 { '+' } else { '-' };
            let mut m = i32::abs(tm.tm_gmtoff) / 60_i32;
            let h = m / 60_i32;
            m -= h * 60_i32;
            fmt!{"%c%02d%02d", sign, h as int, m as int}
          }
          //'+' {}
          '%' => ~"%"
        }
    }

    let mut buf = ~"";

    do io::with_str_reader(format) |rdr| {
        while !rdr.eof() {
            match rdr.read_char() {
                '%' => buf += parse_type(rdr.read_char(), tm),
                ch => str::push_char(buf, ch)
            }
        }
    }

    buf
}

impl tm {
    /// Convert time to the seconds from January 1, 1970
    fn to_timespec() -> timespec {
        let mut sec = 0i64;
        if self.tm_gmtoff == 0_i32 {
            rustrt::rust_timegm(self, sec);
        } else {
            rustrt::rust_mktime(self, sec);
        }
        { sec: sec, nsec: self.tm_nsec }
    }

    /// Convert time to the local timezone
    fn to_local() -> tm {
        at(self.to_timespec())
    }

    /// Convert time to the UTC
    fn to_utc() -> tm {
        at_utc(self.to_timespec())
    }

    /**
     * Return a string of the current time in the form
     * "Thu Jan  1 00:00:00 1970".
     */
    fn ctime() -> ~str { self.strftime(~"%c") }

    /// Formats the time according to the format string.
    fn strftime(format: ~str) -> ~str { strftime(format, self) }

    /**
     * Returns a time string formatted according to RFC 822.
     *
     * local: "Thu, 22 Mar 2012 07:53:18 PST"
     * utc:   "Thu, 22 Mar 2012 14:53:18 UTC"
     */
    fn rfc822() -> ~str {
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
    fn rfc822z() -> ~str {
        self.strftime(~"%a, %d %b %Y %T %z")
    }

    /**
     * Returns a time string formatted according to ISO 8601.
     *
     * local: "2012-02-22T07:53:18-07:00"
     * utc:   "2012-02-22T14:53:18Z"
     */
    fn rfc3339() -> ~str {
        if self.tm_gmtoff == 0_i32 {
            self.strftime(~"%Y-%m-%dT%H:%M:%SZ")
        } else {
            let s = self.strftime(~"%Y-%m-%dT%H:%M:%S");
            let sign = if self.tm_gmtoff > 0_i32 { '+' } else { '-' };
            let mut m = i32::abs(self.tm_gmtoff) / 60_i32;
            let h = m / 60_i32;
            m -= h * 60_i32;
            s + fmt!{"%c%02d:%02d", sign, h as int, m as int}
        }
    }
}

#[cfg(test)]
mod tests {
    import task;

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

        let time = { sec: 1234567890_i64, nsec: 54321_i32 };
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

        let time = { sec: 1234567890_i64, nsec: 54321_i32 };
        let local = at(time);

        error!{"time_at: %?", local};

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
        let zone = local.tm_zone;
        assert zone == ~"PST" || zone == ~"Pacific Standard Time";

        assert local.tm_nsec == 54321_i32;
    }

    #[test]
    fn test_to_timespec() {
        os::setenv(~"TZ", ~"America/Los_Angeles");
        tzset();

        let time = { sec: 1234567890_i64, nsec: 54321_i32 };
        let utc = at_utc(time);

        assert utc.to_timespec() == time;
        assert utc.to_local().to_timespec() == time;
    }

    #[test]
    fn test_conversions() {
        os::setenv(~"TZ", ~"America/Los_Angeles");
        tzset();

        let time = { sec: 1234567890_i64, nsec: 54321_i32 };
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
          ok(tm) => {
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
          err(_) => ()
        }

        let format = ~"%a %b %e %T %Y";
        assert strptime(~"", format) == err(~"Invalid time");
        assert strptime(~"Fri Feb 13 15:31:30", format)
            == err(~"Invalid time");

        match strptime(~"Fri Feb 13 15:31:30 2009", format) {
          err(e) => fail e,
          ok(tm) => {
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

        fn test(s: ~str, format: ~str) -> bool {
            match strptime(s, format) {
              ok(tm) => tm.strftime(format) == s,
              err(e) => fail e
            }
        }

        [
            ~"Sunday",
            ~"Monday",
            ~"Tuesday",
            ~"Wednesday",
            ~"Thursday",
            ~"Friday",
            ~"Saturday"
        ]/_.iter(|day| assert test(day, ~"%A"));

        [
            ~"Sun",
            ~"Mon",
            ~"Tue",
            ~"Wed",
            ~"Thu",
            ~"Fri",
            ~"Sat"
        ]/_.iter(|day| assert test(day, ~"%a"));

        [
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
        ]/_.iter(|day| assert test(day, ~"%B"));

        [
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
        ]/_.iter(|day| assert test(day, ~"%b"));

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
        assert strptime(~"UTC", ~"%Z").get().tm_zone == ~"UTC";
        assert strptime(~"PST", ~"%Z").get().tm_zone == ~"";
        assert strptime(~"-0000", ~"%z").get().tm_gmtoff == 0_i32;
        assert strptime(~"-0800", ~"%z").get().tm_gmtoff == 0_i32;
        assert test(~"%", ~"%%");
    }

    #[test]
    fn test_ctime() {
        os::setenv(~"TZ", ~"America/Los_Angeles");
        tzset();

        let time = { sec: 1234567890_i64, nsec: 54321_i32 };
        let utc   = at_utc(time);
        let local = at(time);

        error!{"test_ctime: %? %?", utc.ctime(), local.ctime()};

        assert utc.ctime()   == ~"Fri Feb 13 23:31:30 2009";
        assert local.ctime() == ~"Fri Feb 13 15:31:30 2009";
    }

    #[test]
    fn test_strftime() {
        os::setenv(~"TZ", ~"America/Los_Angeles");
        tzset();

        let time = { sec: 1234567890_i64, nsec: 54321_i32 };
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
