#![crate_type = "lib"]

extern {
    #[ffi_pure] //~ ERROR the `#[ffi_pure]` attribute is an experimental feature
    pub fn foo();
}
