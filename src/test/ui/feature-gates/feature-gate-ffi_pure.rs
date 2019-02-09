// ignore-tidy-linelength
#![crate_type = "lib"]

extern {
    #[ffi_pure] //~ ERROR the `#[ffi_pure]` attribute is an experimental feature (see issue #58314)
    pub fn foo();
}
