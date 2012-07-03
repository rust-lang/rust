// xfail-test
mod a {
    type rust_task = uint;
    extern mod rustrt {
        fn rust_task_is_unwinding(rt: *rust_task) -> bool;
    }
}

mod b {
    type rust_task = bool;
    extern mod rustrt {
        fn rust_task_is_unwinding(rt: *rust_task) -> bool;
    }
}

fn main() { }
