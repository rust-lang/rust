#![feature(ffi_const)]
#![crate_type = "lib"]

#[unsafe(ffi_const)] //~ ERROR `#[ffi_const]` may only be used on foreign functions
pub fn foo() {}

#[unsafe(ffi_const)] //~ ERROR `#[ffi_const]` may only be used on foreign functions
macro_rules! bar {
    () => {};
}

extern "C" {
    #[unsafe(ffi_const)] //~ ERROR `#[ffi_const]` may only be used on foreign functions
    static INT: i32;

    #[ffi_const] //~ ERROR unsafe attribute used without unsafe
    fn bar();
}
