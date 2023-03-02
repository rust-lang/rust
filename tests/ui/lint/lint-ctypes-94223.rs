#![crate_type = "lib"]
#![deny(improper_ctypes_definitions)]

pub fn bad(f: extern "C" fn([u8])) {}
//~^ ERROR `extern` fn uses type `[u8]`, which is not FFI-safe

pub fn bad_twice(f: Result<extern "C" fn([u8]), extern "C" fn([u8])>) {}
//~^ ERROR `extern` fn uses type `[u8]`, which is not FFI-safe
//~^^ ERROR `extern` fn uses type `[u8]`, which is not FFI-safe
