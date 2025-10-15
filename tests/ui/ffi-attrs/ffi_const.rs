#![feature(ffi_const)]
#![crate_type = "lib"]

#[unsafe(ffi_const)] //~ ERROR attribute cannot be used on
pub fn foo() {}

#[unsafe(ffi_const)] //~ ERROR attribute cannot be used on
macro_rules! bar {
    () => {};
}

extern "C" {
    #[unsafe(ffi_const)] //~ ERROR attribute cannot be used on
    static INT: i32;

    #[ffi_const] //~ ERROR unsafe attribute used without unsafe
    fn bar();
}
