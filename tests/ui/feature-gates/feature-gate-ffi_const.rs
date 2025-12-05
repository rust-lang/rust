#![crate_type = "lib"]

extern "C" {
    #[unsafe(ffi_const)] //~ ERROR the `#[ffi_const]` attribute is an experimental feature
    pub fn foo();
}
