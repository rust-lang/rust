#![feature(ffi_returns_twice)]
#![crate_type = "lib"]

#[ffi_returns_twice] //~ ERROR `#[ffi_returns_twice]` may only be used on foreign functions
pub fn foo() {}

#[ffi_returns_twice] //~ ERROR `#[ffi_returns_twice]` may only be used on foreign functions
macro_rules! bar {
    () => ()
}

extern "C" {
    #[ffi_returns_twice] //~ ERROR `#[ffi_returns_twice]` may only be used on foreign functions
    static INT: i32;
}
