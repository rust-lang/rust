// revisions:cfail1 cfail2
// check-pass
// compile-flags: --crate-type staticlib

#![deny(unused_attributes)]

#[no_mangle]
pub extern "C" fn rust_no_mangle() -> i32 {
    42
}
