//@ compile-flags:--no-defaults --passes strip-priv-imports
//@ aux-build:empty.rs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/27104
#![crate_name="issue_27104"]

//@ has issue_27104/index.html
//@ !hasraw - 'extern crate std'
//@ !hasraw - 'use std::prelude::'

//@ hasraw - 'pub extern crate empty'
pub extern crate empty;
