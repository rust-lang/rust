// revisions:rpass1 rpass2
// compile-flags: --crate-type cdylib
// skip-codegen

#![deny(unused_attributes)]

#[no_mangle]
pub extern "C" fn rust_no_mangle() -> i32 {
    42
}
