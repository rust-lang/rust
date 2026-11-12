#![feature(link_cfg)]
#![crate_type = "staticlib"]

#[link(name = "one", kind = "static", cfg(use_one))]
#[link(name = "does_not_exist", kind = "static", cfg(never_enabled))]
extern "C" {
    fn native_value() -> i32;
}

#[no_mangle]
pub extern "C" fn local_value() -> i32 {
    unsafe { native_value() }
}
