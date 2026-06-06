//@ only-wasm32
//@ build-fail
//@ compile-flags: -C target-feature=-multivalue --crate-type rlib

#![feature(abi_wasm_multivalue)]

#[no_mangle]
pub extern "wasm-multivalue" fn f() {
    //~^ ERROR ABI "wasm-multivalue" which requires the `multivalue` target feature
}

#[no_mangle]
pub fn call_site() {
    f();
    //~^ ERROR ABI "wasm-multivalue" which requires the `multivalue` target feature
}
