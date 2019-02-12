// ignore-tidy-linelength
#![feature(c_ffi_const, c_ffi_pure)]

extern {
    #[c_ffi_pure] //~ ERROR `#[c_ffi_const]` function cannot be`#[c_ffi_pure]` [E0726]
    #[c_ffi_const]
    pub fn baz();
}

fn main() {
    unsafe { baz() };
}
