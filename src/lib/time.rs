native "rust" mod rustrt {
    fn get_time(&mutable u32 sec, &mutable u32 usec);
}

type timeval = rec(u32 sec, u32 usec);

fn get_time() -> timeval {
    auto sec = 0u32; auto usec = 0u32;
    rustrt::get_time(sec, usec);
    ret rec(sec=sec, usec=usec);
}

