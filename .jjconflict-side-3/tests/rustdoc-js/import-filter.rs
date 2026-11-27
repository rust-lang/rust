#![crate_name = "foo"]

pub extern crate std as st2;

pub use crate::Bar as st;

pub struct Bar;
