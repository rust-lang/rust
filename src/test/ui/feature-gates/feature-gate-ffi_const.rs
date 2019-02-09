// ignore-tidy-linelength
#![crate_type = "lib"]

extern {
    #[ffi_const] //~ ERROR the `#[ffi_const]` attribute is an experimental feature (see issue #58314)
    pub fn foo();
}
