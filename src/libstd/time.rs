import libc::{c_char, c_int, c_long, size_t, time_t};
import io::{reader, reader_util};
import result::{result, ok, err, extensions};

export
    timespec,
    get_time,
    precise_time_ns,
    precise_time_s,
    tm,
    empty_tm,
    now,
    at,
    now_utc,
    at_utc;

#[abi = "cdecl"]
native mod rustrt {
    fn get_time(&sec: i64, &nsec: i32);
    fn precise_time_ns(&ns: u64);

    // FIXME: The i64 values can be passed by-val when #2064 is fixed.
    fn rust_gmtime(&&sec: i64, &&nsec: i32, &&result: tm);
    fn rust_localtime(&&sec: i64, &&nsec: i32, &result: tm);
    fn rust_timegm(&&tm: tm, &sec: i64);
    fn rust_mktime(&&tm: tm, &sec: i64);
}

#[doc = "A record specifying a time value in seconds and microseconds."]
type timespec = {sec: i64, nsec: i32};

#[doc = "
Returns the current time as a `timespec` containing the seconds and
microseconds since 1970-01-01T00:00:00Z.
"]
fn get_time() -> timespec {
    let mut sec = 0i64;
    let mut nsec = 0i32;
    rustrt::get_time(sec, nsec);
    ret {sec: sec, nsec: nsec};
}

#[doc = "
Returns the current value of a high-resolution performance counter
in nanoseconds since an unspecified epoch.
"]
fn precise_time_ns() -> u64 {
    let mut ns = 0u64;
    rustrt::precise_time_ns(ns);
    ns
}

#[doc = "
Returns the current value of a high-resolution performance counter
in seconds since an unspecified epoch.
"]
fn precise_time_s() -> float {
    ret (precise_time_ns() as float) / 1000000000.;
}

type tm = {
    tm_sec: i32, // seconds after the minute [0-60]
    tm_min: i32, // minutes after the hour [0-59]
    tm_hour: i32, // hours after midnight [0-23]
    tm_mday: i32, // days of the month [1-31]
    tm_mon: i32, // months since January [0-11]
    tm_year: i32, // years since 1900
    tm_wday: i32, // days since Sunday [0-6]
    tm_yday: i32, // days since January 1 [0-365]
    tm_isdst: i32, // Daylight Savings Time flag
    tm_gmtoff: i32, // offset from UTC in seconds
    tm_zone: str, // timezone abbreviation
    tm_nsec: i32, // nanoseconds
};

fn empty_tm() -> tm {
    {
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
        tm_zone: "",
        tm_nsec: 0_i32,
    }
}

#[doc = "Returns the specified time in UTC"]
fn at_utc(clock: timespec) -> tm {
    let mut {sec, nsec} = clock;
    let mut tm = empty_tm();
    rustrt::rust_gmtime(sec, nsec, tm);
    tm
}

#[doc = "Returns the current time in UTC"]
fn now_utc() -> tm {
    at_utc(get_time())
}

#[doc = "Returns the specified time in the local timezone"]
fn at(clock: timespec) -> tm {
    let mut {sec, nsec} = clock;
    let mut tm = empty_tm();
    rustrt::rust_localtime(sec, nsec, tm);
    tm
}

#[doc = "Returns the current time in the local timezone"]
fn now() -> tm {
    at(get_time())
}

fn strftime(format: str, tm: tm) -> str {
    fn parse_type(ch: char, tm: tm) -> str {
        //FIXME: Implement missing types.
        alt check ch {
          'A' {
            alt check tm.tm_wday as int {
              0 { "Sunday" }
              1 { "Monday" }
              2 { "Tuesday" }
              3 { "Wednesday" }
              4 { "Thursday" }
              5 { "Friday" }
              6 { "Saturday" }
            }
          }
          'a' {
            alt check tm.tm_wday as int {
              0 { "Sun" }
              1 { "Mon" }
              2 { "Tue" }
              3 { "Wed" }
              4 { "Thu" }
              5 { "Fri" }
              6 { "Sat" }
            }
          }
          'B' {
            alt check tm.tm_mon as int {
              0 { "January" }
              1 { "February" }
              2 { "March" }
              3 { "April" }
              4 { "May" }
              5 { "June" }
              6 { "July" }
              7 { "August" }
              8 { "September" }
              9 { "October" }
              10 { "November" }
              11 { "December" }
            }
          }
          'b' | 'h' {
            alt check tm.tm_mon as int {
              0 { "Jan" }
              1 { "Feb" }
              2 { "Mar" }
              3 { "Apr" }
              4 { "May" }
              5 { "Jun" }
              6 { "Jul" }
              7 { "Aug" }
              8 { "Sep" }
              9 { "Oct" }
              10 { "Nov" }
              11 { "Dec" }
            }
          }
          'C' { #fmt("%02d", (tm.tm_year as int + 1900) / 100) }
          'c' {
            #fmt("%s %s %s %s %s",
                parse_type('a', tm),
                parse_type('b', tm),
                parse_type('e', tm),
                parse_type('T', tm),
                parse_type('Y', tm))
          }
          'D' | 'x' {
            #fmt("%s/%s/%s",
                parse_type('m', tm),
                parse_type('d', tm),
                parse_type('y', tm))
          }
          'd' { #fmt("%02d", tm.tm_mday as int) }
          'e' { #fmt("%2d", tm.tm_mday as int) }
          'F' {
            #fmt("%s-%s-%s",
                parse_type('Y', tm),
                parse_type('m', tm),
                parse_type('d', tm))
          }
          //'G' {}
          //'g' {}
          'H' { #fmt("%02d", tm.tm_hour as int) }
          'I' {
            let mut h = tm.tm_hour as int;
            if h == 0 { h = 12 }
            if h > 12 { h -= 12 }
            #fmt("%02d", h)
          }
          'j' { #fmt("%03d", tm.tm_yday as int + 1) }
          'k' { #fmt("%2d", tm.tm_hour as int) }
          'l' {
            let mut h = tm.tm_hour as int;
            if h == 0 { h = 12 }
            if h > 12 { h -= 12 }
            #fmt("%2d", h)
          }
          'M' { #fmt("%02d", tm.tm_min as int) }
          'm' { #fmt("%02d", tm.tm_mon as int + 1) }
          'n' { "\n" }
          'P' { if tm.tm_hour as int < 12 { "am" } else { "pm" } }
          'p' { if tm.tm_hour as int < 12 { "AM" } else { "PM" } }
          'R' {
            #fmt("%s:%s",
                parse_type('H', tm),
                parse_type('M', tm))
          }
          'r' {
            #fmt("%s:%s:%s %s",
                parse_type('I', tm),
                parse_type('M', tm),
                parse_type('S', tm),
                parse_type('p', tm))
          }
          'S' { #fmt("%02d", tm.tm_sec as int) }
          's' { #fmt("%d", tm.to_timespec().sec as int) }
          'T' | 'X' {
            #fmt("%s:%s:%s",
                parse_type('H', tm),
                parse_type('M', tm),
                parse_type('S', tm))
          }
          't' { "\t" }
          //'U' {}
          'u' {
            let i = tm.tm_wday as int;
            int::str(if i == 0 { 7 } else { i })
          }
          //'V' {}
          'v' {
            #fmt("%s-%s-%s",
                parse_type('e', tm),
                parse_type('b', tm),
                parse_type('Y', tm))
          }
          //'W' {}
          'w' { int::str(tm.tm_wday as int) }
          //'X' {}
          //'x' {}
          'Y' { int::str(tm.tm_year as int + 1900) }
          'y' { #fmt("%02d", (tm.tm_year as int + 1900) % 100) }
          'Z' { tm.tm_zone }
          'z' {
            let sign = if tm.tm_gmtoff > 0_i32 { '+' } else { '-' };
            let mut m = i32::abs(tm.tm_gmtoff) / 60_i32;
            let h = m / 60_i32;
            m -= h * 60_i32;
            #fmt("%c%02d%02d", sign, h as int, m as int)
          }
          //'+' {}
          '%' { "%" }
        }
    }

    let mut buf = "";

    io::with_str_reader(format) { |rdr|
        while !rdr.eof() {
            alt rdr.read_char() {
                '%' { buf += parse_type(rdr.read_char(), tm); }
                ch { str::push_char(buf, ch); }
            }
        }
    }

    buf
}

impl tm for tm {
    #[doc = "Convert time to the seconds from January 1, 1970"]
    fn to_timespec() -> timespec {
        let mut sec = 0i64;
        if self.tm_gmtoff == 0_i32 {
            rustrt::rust_timegm(self, sec);
        } else {
            rustrt::rust_mktime(self, sec);
        }
        { sec: sec, nsec: self.tm_nsec }
    }

    #[doc = "Convert time to the local timezone"]
    fn to_local() -> tm {
        at(self.to_timespec())
    }

    #[doc = "Convert time to the UTC"]
    fn to_utc() -> tm {
        at_utc(self.to_timespec())
    }

    #[doc = "
    Return a string of the current time in the form
    \"Thu Jan  1 00:00:00 1970\".
    "]
    fn ctime() -> str { self.strftime("%c") }

    #[doc = "Formats the time according to the format string."]
    fn strftime(format: str) -> str { strftime(format, self) }

    #[doc = "
    Returns a time string formatted according to RFC 822.

    local: \"Thu, 22 Mar 2012 07:53:18 PST\"
    utc:   \"Thu, 22 Mar 2012 14:53:18 UTC\"
    "]
    fn rfc822() -> str {
        if self.tm_gmtoff == 0_i32 {
            self.strftime("%a, %d %b %Y %T GMT")
        } else {
            self.strftime("%a, %d %b %Y %T %Z")
        }
    }

    #[doc = "
    Returns a time string formatted according to RFC 822 with Zulu time.

    local: \"Thu, 22 Mar 2012 07:53:18 -0700\"
    utc:   \"Thu, 22 Mar 2012 14:53:18 -0000\"
    "]
    fn rfc822z() -> str {
        self.strftime("%a, %d %b %Y %T %z")
    }

    #[doc = "
    Returns a time string formatted according to ISO 8601.

    local: \"2012-02-22T07:53:18-07:00\"
    utc:   \"2012-02-22T14:53:18Z\"
    "]
    fn rfc3339() -> str {
        if self.tm_gmtoff == 0_i32 {
            self.strftime("%Y-%m-%dT%H:%M:%SZ")
        } else {
            let s = self.strftime("%Y-%m-%dT%H:%M:%S");
            let sign = if self.tm_gmtoff > 0_i32 { '+' } else { '-' };
            let mut m = i32::abs(self.tm_gmtoff) / 60_i32;
            let h = m / 60_i32;
            m -= h * 60_i32;
            s + #fmt("%c%02d:%02d", sign, h as int, m as int)
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
        log(debug, "tv1=" + uint::str(tv1.sec as uint) + " sec + "
                   + uint::str(tv1.nsec as uint) + " nsec");

        assert tv1.sec > some_recent_date;
        assert tv1.nsec < 1000000000i32;

        let tv2 = get_time();
        log(debug, "tv2=" + uint::str(tv2.sec as uint) + " sec + "
                   + uint::str(tv2.nsec as uint) + " nsec");

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

        log(debug, "s0=" + float::to_str(s0, 9u) + " sec");
        assert s0 > 0.;
        let ns0 = (s0 * 1000000000.) as u64;
        log(debug, "ns0=" + u64::str(ns0) + " ns");

        log(debug, "ns1=" + u64::str(ns1) + " ns");
        assert ns1 >= ns0;

        let ns2 = precise_time_ns();
        log(debug, "ns2=" + u64::str(ns2) + " ns");
        assert ns2 >= ns1;
    }

    #[test]
    fn test_at_utc() {
        os::setenv("TZ", "America/Los_Angeles");

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
        assert utc.tm_zone == "UTC";
        assert utc.tm_nsec == 54321_i32;
    }

    #[test]
    fn test_at() {
        os::setenv("TZ", "America/Los_Angeles");

        let time = { sec: 1234567890_i64, nsec: 54321_i32 };
        let local = at(time);

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

        // FIXME: We should probably standardize on the timezone
        // abbreviation.
        let zone = local.tm_zone;
        assert zone == "PST" || zone == "Pacific Standard Time";

        assert local.tm_nsec == 54321_i32;
    }

    #[test]
    fn test_to_timespec() {
        os::setenv("TZ", "America/Los_Angeles");

        let time = { sec: 1234567890_i64, nsec: 54321_i32 };
        let utc = at_utc(time);

        assert utc.to_timespec() == time;
        assert utc.to_local().to_timespec() == time;
    }

    #[test]
    fn test_conversions() {
        os::setenv("TZ", "America/Los_Angeles");

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
    fn test_ctime() {
        os::setenv("TZ", "America/Los_Angeles");

        let time = { sec: 1234567890_i64, nsec: 54321_i32 };
        let utc   = at_utc(time);
        let local = at(time);

        assert utc.ctime()   == "Fri Feb 13 23:31:30 2009";
        assert local.ctime() == "Fri Feb 13 15:31:30 2009";
    }

    #[test]
    fn test_strftime() {
        os::setenv("TZ", "America/Los_Angeles");

        let time = { sec: 1234567890_i64, nsec: 54321_i32 };
        let utc = at_utc(time);
        let local = at(time);

        assert local.strftime("") == "";
        assert local.strftime("%A") == "Friday";
        assert local.strftime("%a") == "Fri";
        assert local.strftime("%B") == "February";
        assert local.strftime("%b") == "Feb";
        assert local.strftime("%C") == "20";
        assert local.strftime("%c") == "Fri Feb 13 15:31:30 2009";
        assert local.strftime("%D") == "02/13/09";
        assert local.strftime("%d") == "13";
        assert local.strftime("%e") == "13";
        assert local.strftime("%F") == "2009-02-13";
        // assert local.strftime("%G") == "2009";
        // assert local.strftime("%g") == "09";
        assert local.strftime("%H") == "15";
        assert local.strftime("%I") == "03";
        assert local.strftime("%j") == "044";
        assert local.strftime("%k") == "15";
        assert local.strftime("%l") == " 3";
        assert local.strftime("%M") == "31";
        assert local.strftime("%m") == "02";
        assert local.strftime("%n") == "\n";
        assert local.strftime("%P") == "pm";
        assert local.strftime("%p") == "PM";
        assert local.strftime("%R") == "15:31";
        assert local.strftime("%r") == "03:31:30 PM";
        assert local.strftime("%S") == "30";
        assert local.strftime("%s") == "1234567890";
        assert local.strftime("%T") == "15:31:30";
        assert local.strftime("%t") == "\t";
        // assert local.strftime("%U") == "06";
        assert local.strftime("%u") == "5";
        // assert local.strftime("%V") == "07";
        assert local.strftime("%v") == "13-Feb-2009";
        // assert local.strftime("%W") == "06";
        assert local.strftime("%w") == "5";
        // handle "%X"
        // handle "%x"
        assert local.strftime("%Y") == "2009";
        assert local.strftime("%y") == "09";

        // FIXME: We should probably standardize on the timezone
        // abbreviation.
        let zone = local.strftime("%Z");
        assert zone == "PST" || zone == "Pacific Standard Time";

        assert local.strftime("%z") == "-0800";
        assert local.strftime("%%") == "%";

        // FIXME: We should probably standardize on the timezone
        // abbreviation.
        let rfc822 = local.rfc822();
        let prefix = "Fri, 13 Feb 2009 15:31:30 ";
        assert rfc822 == prefix + "PST" ||
               rfc822 == prefix + "Pacific Standard Time";

        assert local.ctime() == "Fri Feb 13 15:31:30 2009";
        assert local.rfc822z() == "Fri, 13 Feb 2009 15:31:30 -0800";
        assert local.rfc3339() == "2009-02-13T15:31:30-08:00";

        assert utc.ctime() == "Fri Feb 13 23:31:30 2009";
        assert utc.rfc822() == "Fri, 13 Feb 2009 23:31:30 GMT";
        assert utc.rfc822z() == "Fri, 13 Feb 2009 23:31:30 -0000";
        assert utc.rfc3339() == "2009-02-13T23:31:30Z";
    }
}
