/*
Can we bind native things?
*/

native "c-stack-cdecl" mod rustrt {
    fn task_yield();
}

fn main() { bind rustrt::task_yield(); }
