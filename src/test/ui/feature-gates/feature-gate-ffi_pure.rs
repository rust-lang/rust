// ignore-tidy-linelength
#![crate_type = "lib"]

extern {
    #[ffi_pure] //~ ERROR the `#[ffi_pure]` attribute is an experimental feature (see issue #58329)
    pub fn foo();
}
