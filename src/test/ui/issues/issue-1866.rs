// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#![allow(non_camel_case_types)]

// pretty-expanded FIXME #23616

mod a {
    pub type rust_task = usize;
    pub mod rustrt {
        use super::rust_task;
        extern {
            pub fn rust_task_is_unwinding(rt: *const rust_task) -> bool;
        }
    }
}

mod b {
    pub type rust_task = bool;
    pub mod rustrt {
        use super::rust_task;
        extern {
            pub fn rust_task_is_unwinding(rt: *const rust_task) -> bool;
        }
    }
}

pub fn main() { }
