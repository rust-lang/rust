//@ edition:2018

// https://github.com/rust-lang/rust/issues/89852
#![crate_name = "foo"]
#![no_core]
#![feature(no_core)]

//@ matchesraw 'foo/sidebar-items.js' '"repro"'
//@ !matchesraw 'foo/sidebar-items.js' '"repro".*"repro"'

#[macro_export]
macro_rules! repro {
    () => {};
}

pub use crate::repro as repro2;
