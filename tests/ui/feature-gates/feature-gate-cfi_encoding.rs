#![crate_type = "lib"]

#[cfi_encoding = "3Bar"] //~ ERROR the `#[cfi_encoding]` attribute is an experimental feature [E0658]
pub struct Foo(i32);
