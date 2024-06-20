#![crate_type = "dylib"]

extern crate an_rlib;

// This should be exported
pub fn public_rust_function_from_rust_dylib() {}

// This should be exported
#[no_mangle]
pub extern "C" fn public_c_function_from_rust_dylib() {
    let _ = public_generic_function_from_rust_dylib(1u16);
}

// This should be exported if -Zshare-generics=yes
pub fn public_generic_function_from_rust_dylib<T>(x: T) -> T {
    x
}
