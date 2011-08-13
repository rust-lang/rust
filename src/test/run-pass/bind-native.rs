/*
Can we bind native things?
*/

//xfail-stage0
//xfail-stage1
//xfail-stage2
//xfail-stage3

native "rust" mod rustrt {
    fn task_yield();
}

fn main() { bind rustrt::task_yield(); }
