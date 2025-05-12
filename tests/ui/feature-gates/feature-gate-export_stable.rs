#![crate_type="lib"]

#[export_stable]
//~^ ERROR the `#[export_stable]` attribute is an experimental feature
pub mod a {}
