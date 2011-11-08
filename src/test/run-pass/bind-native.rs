/*
Can we bind native things?
*/

native "c-stack-cdecl" mod rustrt {
    fn pin_task();
}

fn main() { bind rustrt::pin_task(); }
