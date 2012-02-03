// xfail-fast - Somehow causes check-fast to livelock?? Probably because we're
// calling pin_task and that's having wierd side-effects.

#[abi = "cdecl"]
#[link_name = "rustrt"]
native mod rustrt1 {
    fn last_os_error() -> str;
}

#[abi = "cdecl"]
#[link_name = "rustrt"]
native mod rustrt2 {
    fn last_os_error() -> str;
}

fn main() {
    rustrt1::last_os_error();
    rustrt2::last_os_error();
}
