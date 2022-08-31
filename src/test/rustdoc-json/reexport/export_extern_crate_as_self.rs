//! Regression test for <https://github.com/rust-lang/rust/issues/100531>

#![feature(no_core)]
#![no_core]

#![crate_name = "export_extern_crate_as_self"]

// ignore-tidy-linelength

// @is "$.index[*][?(@.kind=='module')].name" \"export_extern_crate_as_self\"
pub extern crate self as export_extern_crate_as_self; // Must be the same name as the crate already has
