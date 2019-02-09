// ignore-tidy-linelength
#![feature(ffi_pure)]
#![crate_type = "lib"]

#[ffi_pure] //~ ERROR `#[ffi_pure]` may only be used on `extern fn`s [E0724]
pub fn foo() {}
