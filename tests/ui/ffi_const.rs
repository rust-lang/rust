#![feature(ffi_const)]
#![crate_type = "lib"]

#[ffi_const] //~ ERROR `#[ffi_const]` may only be used on foreign functions
pub fn foo() {}

#[ffi_const] //~ ERROR `#[ffi_const]` may only be used on foreign functions
macro_rules! bar {
    () => ()
}

extern "C" {
    #[ffi_const] //~ ERROR `#[ffi_const]` may only be used on foreign functions
    static INT: i32;
}
