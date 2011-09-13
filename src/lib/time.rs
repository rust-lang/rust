

native "rust" mod rustrt {
    fn get_time(&sec: u32, &usec: u32);
    fn nano_time(&ns: u64);
}

type timeval = {sec: u32, usec: u32};

fn get_time() -> timeval {
    let sec = 0u32;
    let usec = 0u32;
    rustrt::get_time(sec, usec);
    ret {sec: sec, usec: usec};
}

fn precise_time_ns() -> u64 { let ns = 0u64; rustrt::nano_time(ns); ret ns; }

fn precise_time_s() -> float {
    ret (precise_time_ns() as float) / 1000000000.;
}
