

native "rust" mod rustrt {
    fn get_time(&mutable u32 sec, &mutable u32 usec);
    fn nano_time(&mutable u64 ns);
}

type timeval = rec(u32 sec, u32 usec);

fn get_time() -> timeval {
    auto sec = 0u32;
    auto usec = 0u32;
    rustrt::get_time(sec, usec);
    ret rec(sec=sec, usec=usec);
}

fn precise_time_ns() -> u64 {
    auto ns = 0u64;
    rustrt::nano_time(ns);
    ret ns;
}

fn precise_time_s() -> float {
    ret (precise_time_ns() as float) / 1000000000.;
}
