#[abi = "cdecl"]
native mod rustrt {
    fn get_time(&sec: u32, &usec: u32);
    fn precise_time_ns(&ns: u64);
}

#[doc = "A record specifying a time value in seconds and microseconds."]
type timeval = {sec: u32, usec: u32};

#[doc = "
Returns the current time as a `timeval` containing the seconds and
microseconds since 1970-01-01T00:00:00Z.
"]
fn get_time() -> timeval {
    let sec = 0u32;
    let usec = 0u32;
    rustrt::get_time(sec, usec);
    ret {sec: sec, usec: usec};
}

#[doc = "
Returns the current value of a high-resolution performance counter
in nanoseconds since an unspecified epoch.
"]
fn precise_time_ns() -> u64 { let ns = 0u64; rustrt::precise_time_ns(ns); ns }

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
        const some_recent_date: u32 = 1325376000u32; // 2012-01-01T00:00:00Z
        const some_future_date: u32 = 1577836800u32; // 2020-01-01T00:00:00Z

        let tv1 = get_time();
        log(debug, "tv1=" + uint::str(tv1.sec as uint) + " sec + "
                   + uint::str(tv1.usec as uint) + " usec");

        assert tv1.sec > some_recent_date;
        assert tv1.usec < 1000000u32;

        let tv2 = get_time();
        log(debug, "tv2=" + uint::str(tv2.sec as uint) + " sec + "
                   + uint::str(tv2.usec as uint) + " usec");

        assert tv2.sec >= tv1.sec;
        assert tv2.sec < some_future_date;
        assert tv2.usec < 1000000u32;
        if tv2.sec == tv1.sec {
            assert tv2.usec >= tv1.usec;
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
