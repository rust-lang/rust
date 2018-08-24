// aux-build:macro_crate_test.rs
// ignore-cross-compile
//
// macro_crate_test will not compile on a cross-compiled target because
// libsyntax is not compiled for it.

#![deny(plugin_as_library)]
#![allow(unused_extern_crates)]

extern crate macro_crate_test; //~ ERROR compiler plugin used as an ordinary library

fn main() { }
