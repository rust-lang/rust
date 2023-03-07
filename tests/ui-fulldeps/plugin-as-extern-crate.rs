// check-pass
// aux-build:empty-plugin.rs
// ignore-cross-compile
//
// empty_plugin will not compile on a cross-compiled target because
// librustc_ast is not compiled for it.

extern crate empty_plugin; // OK, plugin crates are still crates

fn main() {}
