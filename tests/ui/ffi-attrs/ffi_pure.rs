#![feature(ffi_pure)]
#![crate_type = "lib"]

#[unsafe(ffi_pure)] //~ ERROR `#[ffi_pure]` may only be used on foreign functions
pub fn foo() {}

#[unsafe(ffi_pure)] //~ ERROR `#[ffi_pure]` may only be used on foreign functions
macro_rules! bar {
    () => {};
}

extern "C" {
    #[unsafe(ffi_pure)] //~ ERROR `#[ffi_pure]` may only be used on foreign functions
    static INT: i32;

    #[ffi_pure] //~ ERROR unsafe attribute used without unsafe
    fn bar();
}
