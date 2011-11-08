/*
Can we bind native things?
*/

native "c-stack-cdecl" mod rustrt {
    fn task_sleep();
}

fn main() { bind rustrt::task_sleep(); }
