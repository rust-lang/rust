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
}
