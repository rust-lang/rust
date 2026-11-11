//@ revisions: cpass1 cpass2
//@ compile-flags: --crate-type cdylib
//@ needs-crate-type: cdylib

#![deny(unused_attributes)]

#[no_mangle]
pub extern "C" fn rust_no_mangle() -> i32 {
    42
}
