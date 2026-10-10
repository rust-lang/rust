#![crate_type = "proc-macro"]

extern crate an_rlib;

// This should not be exported
#[no_mangle]
extern "C" fn public_c_function_from_cdylib() {
    an_rlib::public_c_function_from_rlib();
}
