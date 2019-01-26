// aux-build:basic_plugin.rs
// ignore-cross-compile
//
// basic_plugin will not compile on a cross-compiled target because
// libsyntax is not compiled for it.

#![deny(plugin_as_library)]
#![allow(unused_extern_crates)]

extern crate basic_plugin; //~ ERROR compiler plugin used as an ordinary library

fn main() { }
