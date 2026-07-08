#![feature(linkage)]
#![crate_type = "cdylib"]

#[linkage = "weak"]
#[no_mangle]
pub fn weak_function() {}

#[linkage = "weak"]
#[no_mangle]
pub static WEAK_STATIC: u8 = 42;
