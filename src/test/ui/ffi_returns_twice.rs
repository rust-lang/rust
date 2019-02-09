// ignore-tidy-linelength
#![feature(ffi_returns_twice)]
#![crate_type = "lib"]

#[ffi_returns_twice] //~ ERROR `#[ffi_returns_twice]` may only be used on `extern fn`s
pub fn foo() {}
