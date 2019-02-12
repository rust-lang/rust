// ignore-tidy-linelength
#![feature(c_ffi_const, c_ffi_pure)]
#![crate_type = "lib"]

#[c_ffi_const] //~ ERROR `#[c_ffi_const]` may only be used on foreign functions [E0725]
pub fn foo() {}
