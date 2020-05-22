#![feature(ffi_const, ffi_pure)]

extern {
    #[ffi_pure] //~ ERROR `#[ffi_const]` function cannot be `#[ffi_pure]`
    #[ffi_const]
    pub fn baz();
}

fn main() {
    unsafe { baz() };
}
