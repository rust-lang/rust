#![crate_type = "staticlib"]

extern crate dep;

#[no_mangle]
pub extern "C" fn upstream_value() -> i32 {
    dep::value()
}
