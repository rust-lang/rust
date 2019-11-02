// edition:2018
// aux-build:attr-plugin-test.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(attr_plugin_test)]
//~^ WARN use of deprecated attribute `plugin`

pub use mac as reexport; //~ ERROR `mac` is private, and cannot be re-exported

fn main() {}
