// edition:2018
// ignore-stage1
// aux-build:attr-plugin-test.rs

#![feature(plugin)]
#![plugin(attr_plugin_test)]

use empty as full; //~ ERROR cannot import a built-in macro

fn main() {}
