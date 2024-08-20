#![crate_name = "foo"]
#![crate_type = "rlib"]

extern crate dependency;
pub use dependency::Type;
