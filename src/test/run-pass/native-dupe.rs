// xfail-fast - Somehow causes check-fast to livelock?? Probably because we're
// calling pin_task and that's having wierd side-effects.

#[abi = "cdecl"]
#[link_name = "rustrt"]
native mod rustrt1 {
    fn pin_task();
}

#[abi = "cdecl"]
#[link_name = "rustrt"]
native mod rustrt2 {
    fn pin_task();
}

fn main() {
    rustrt1::pin_task();
    rustrt2::pin_task();
}
