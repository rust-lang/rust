/*
Module: time
*/

// FIXME: Document what these functions do

#[abi = "cdecl"]
native mod rustrt {
    fn get_time(&sec: u32, &usec: u32);
    fn nano_time(&ns: u64);
}

/* Type: timeval */
type timeval = {sec: u32, usec: u32};

/* Function: get_time */
fn get_time() -> timeval {
    let sec = 0u32;
    let usec = 0u32;
    rustrt::get_time(sec, usec);
    ret {sec: sec, usec: usec};
}

/* Function: precise_time_ns */
fn precise_time_ns() -> u64 { let ns = 0u64; rustrt::nano_time(ns); ret ns; }

/* Function: precise_time_s */
fn precise_time_s() -> float {
    ret (precise_time_ns() as float) / 1000000000.;
}
