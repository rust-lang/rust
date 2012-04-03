#[abi = "cdecl"]
native mod rustrt {
    fn get_time(&sec: i64, &nsec: i32);
    fn precise_time_ns(&ns: u64);
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
}
