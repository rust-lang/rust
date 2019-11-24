// check-pass
// aux-build:empty-plugin.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(empty_plugin)] //~ WARNING compiler plugins are deprecated

fn main() {}
