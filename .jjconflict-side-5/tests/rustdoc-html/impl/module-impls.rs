#![crate_name = "foo"]

pub use std::marker::Send;

//@ !hasraw foo/index.html 'Implementations'
