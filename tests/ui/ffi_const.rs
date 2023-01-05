#![feature(ffi_const)]
#![crate_type = "lib"]

#[ffi_const] //~ ERROR `#[ffi_const]` may only be used on foreign functions
pub fn foo() {}
