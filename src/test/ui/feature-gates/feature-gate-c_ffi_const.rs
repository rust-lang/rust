// ignore-tidy-linelength
#![crate_type = "lib"]

extern {
    #[c_ffi_const] //~ ERROR the `#[c_ffi_const]` attribute is an experimental feature (see issue #58328)
    pub fn foo();
}
