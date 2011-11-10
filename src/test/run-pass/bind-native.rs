/*
Can we bind native things?
*/

native "cdecl" mod rustrt {
    fn pin_task();
}

fn main() { bind rustrt::pin_task(); }
