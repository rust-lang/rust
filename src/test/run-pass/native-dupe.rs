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
