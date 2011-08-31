/*
Can we bind native things?
*/

//xfail-stage0
//xfail-test

native "rust" mod rustrt {
    fn task_yield();
}

fn main() { bind rustrt::task_yield(); }
