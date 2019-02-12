// ignore-tidy-linelength
#![feature(c_ffi_pure)]
#![crate_type = "lib"]

#[c_ffi_pure] //~ ERROR `#[c_ffi_pure]` may only be used on foreign functions [E0724]
pub fn foo() {}
