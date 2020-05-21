#![feature(ffi_pure)]
#![crate_type = "lib"]

#[ffi_pure] //~ ERROR `#[ffi_pure]` may only be used on foreign functions
pub fn foo() {}
