// xfail-fast - Somehow causes check-fast to livelock?? Probably because we're
// calling pin_task and that's having wierd side-effects.

#[abi = "cdecl"]
#[link_name = "rustrt"]
native mod rustrt1 {
    fn do_gc();
}

#[abi = "cdecl"]
#[link_name = "rustrt"]
native mod rustrt2 {
    fn do_gc();
}

fn main() {
    rustrt1::do_gc();
    rustrt2::do_gc();
}
