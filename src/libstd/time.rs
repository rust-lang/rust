/*
Module: time
*/

#[abi = "cdecl"]
native mod rustrt {
    fn get_time(&sec: u32, &usec: u32);
    fn nano_time(&ns: u64);
}

/*
Type: timeval

A record specifying a time value in seconds and microseconds.
*/
type timeval = {sec: u32, usec: u32};

/*
Function: get_time

Returns the current time as a `timeval` containing the seconds and
microseconds since 1970-01-01T00:00:00Z.
*/
fn get_time() -> timeval {
    let sec = 0u32;
    let usec = 0u32;
    rustrt::get_time(sec, usec);
    ret {sec: sec, usec: usec};
}

/*
Function: precise_time_ns

Returns the current value of a high-resolution performance counter
in nanoseconds since an unspecified epoch.
*/
fn precise_time_ns() -> u64 { let ns = 0u64; rustrt::nano_time(ns); ret ns; }

/*
Function: precise_time_s

Returns the current value of a high-resolution performance counter
in seconds since an unspecified epoch.
*/
fn precise_time_s() -> float {
    ret (precise_time_ns() as float) / 1000000000.;
}
