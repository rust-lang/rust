// ignore-tidy-linelength
#![crate_type = "lib"]

extern {
    #[c_ffi_pure] //~ ERROR the `#[c_ffi_pure]` attribute is an experimental feature (see issue #58329)
    pub fn foo();
}
