#![crate_type = "dylib"]

extern crate bbb;

pub fn do_work() {
    unsafe { bbb::native_func(); }
    bbb::wrapped_func();
}

pub fn do_work_generic<T>() {
    unsafe { bbb::native_func(); }
    bbb::wrapped_func();
}
