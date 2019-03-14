// aux-build:attr-plugin-test.rs
// ignore-cross-compile
//
// attr_plugin_test will not compile on a cross-compiled target because
// libsyntax is not compiled for it.

#![deny(plugin_as_library)]
#![allow(unused_extern_crates)]

extern crate attr_plugin_test; //~ ERROR compiler plugin used as an ordinary library

fn main() { }
