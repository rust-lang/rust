// xfail-fast - Somehow causes check-fast to livelock?? Probably because we're
// calling pin_task and that's having wierd side-effects.

native "cdecl" mod rustrt1 = "rustrt" {
    fn pin_task();
}

native "cdecl" mod rustrt2 = "rustrt" {
    fn pin_task();
}

fn main() {
    rustrt1::pin_task();
    rustrt2::pin_task();
}
