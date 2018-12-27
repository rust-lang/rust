#![crate_type="cdylib"]

extern crate an_rlib;

// This should not be exported
pub fn public_rust_function_from_cdylib() {}

// This should be exported
#[no_mangle]
pub extern "C" fn public_c_function_from_cdylib() {
    an_rlib::public_c_function_from_rlib();
}
