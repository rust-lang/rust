native "rust" mod rustrt {
    fn get_time(&mutable u32 sec, &mutable u32 usec);
}

type timeval = rec(u32 sec, u32 usec);

fn get_time() -> timeval {
    let timeval res = rec(sec=0u32, usec=0u32);
    rustrt::get_time(res.sec, res.usec);
    ret res;
}

