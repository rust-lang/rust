#![crate_type = "lib"]

extern {
    #[ffi_const] //~ ERROR the `#[ffi_const]` attribute is an experimental feature
    pub fn foo();
}
