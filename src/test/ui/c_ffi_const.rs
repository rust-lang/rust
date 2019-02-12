// ignore-tidy-linelength
#![feature(c_ffi_const, c_ffi_pure)]
#![crate_type = "lib"]

#[c_ffi_const] //~ ERROR `#[c_ffi_const]` may only be used on foreign functions [E0725]
pub fn foo() {}

#[c_ffi_pure]
#[c_ffi_const] //~ ERROR `#[c_ffi_const]` functions cannot be `#[c_ffi_pure]` [E0726]
pub fn bar() {}
