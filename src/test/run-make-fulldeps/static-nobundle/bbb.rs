#![crate_type = "rlib"]
#![feature(static_nobundle)]

#[link(name = "aaa", kind = "static-nobundle")]
extern "C" {
    pub fn native_func();
}

pub fn wrapped_func() {
    unsafe {
        native_func();
    }
}
