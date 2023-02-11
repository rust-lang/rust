#![crate_type = "lib"]

extern "C" {
    #[ffi_returns_twice] //~ ERROR the `#[ffi_returns_twice]` attribute is an experimental feature
    pub fn foo();
}
