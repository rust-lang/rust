#![feature(ffi_pure)]
#![crate_type = "lib"]

#[ffi_pure] //~ ERROR `#[ffi_pure]` may only be used on foreign functions
pub fn foo() {}

#[ffi_pure] //~ ERROR `#[ffi_pure]` may only be used on foreign functions
macro_rules! bar {
    () => ()
}

extern "C" {
    #[ffi_pure] //~ ERROR `#[ffi_pure]` may only be used on foreign functions
    static INT: i32;
}
