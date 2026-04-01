#![feature(ffi_pure)]
#![crate_type = "lib"]

#[unsafe(ffi_pure)] //~ ERROR attribute cannot be used on
pub fn foo() {}

#[unsafe(ffi_pure)] //~ ERROR attribute cannot be used on
macro_rules! bar {
    () => {};
}

extern "C" {
    #[unsafe(ffi_pure)] //~ ERROR attribute cannot be used on
    static INT: i32;

    #[ffi_pure] //~ ERROR unsafe attribute used without unsafe
    fn bar();
}
