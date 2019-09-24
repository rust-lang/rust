// edition:2018
// aux-build:attr-plugin-test.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(attr_plugin_test)]

pub use mac as reexport; //~ ERROR `mac` is private, and cannot be re-exported

fn main() {}
