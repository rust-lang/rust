// xfail-test
mod a {
    #[legacy_exports];
    type rust_task = uint;
    extern mod rustrt {
        #[legacy_exports];
        fn rust_task_is_unwinding(rt: *rust_task) -> bool;
    }
}

mod b {
    #[legacy_exports];
    type rust_task = bool;
    extern mod rustrt {
        #[legacy_exports];
        fn rust_task_is_unwinding(rt: *rust_task) -> bool;
    }
}

fn main() { }
