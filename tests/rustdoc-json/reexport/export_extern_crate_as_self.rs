//! Regression test for <https://github.com/rust-lang/rust/issues/100531>

#![crate_name = "export_extern_crate_as_self"]

//@ is "$.index[?(@.inner.module)].name" \"export_extern_crate_as_self\"
pub extern crate self as export_extern_crate_as_self; // Must be the same name as the crate already has
