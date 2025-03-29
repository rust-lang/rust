#![crate_type="dylib"]

#[export]
//~^ ERROR the `#[export]` attribute is an experimental feature
pub mod a {}
