#![crate_name = "foo"]

pub use std::marker::Send;

// @!has foo/index.html 'Implementations'
