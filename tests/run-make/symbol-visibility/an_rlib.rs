#![crate_type = "rlib"]

pub fn public_rust_function_from_rlib() {}

#[no_mangle]
pub extern "C" fn public_c_function_from_rlib() {
    let _ = public_generic_function_from_rlib(0u64);
}

pub fn public_generic_function_from_rlib<T>(x: T) -> T {
    x
}
