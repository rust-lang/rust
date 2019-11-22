// aux-build:empty-plugin.rs
// ignore-cross-compile
//
// empty_plugin will not compile on a cross-compiled target because
// libsyntax is not compiled for it.

#![deny(plugin_as_library)]

extern crate empty_plugin; //~ ERROR compiler plugin used as an ordinary library

fn main() { }
