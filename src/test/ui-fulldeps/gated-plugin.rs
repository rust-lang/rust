// aux-build:empty-plugin.rs
// ignore-stage1

#![plugin(empty_plugin)]
//~^ ERROR compiler plugins are deprecated
//~| WARN use of deprecated attribute `plugin`: compiler plugins are deprecated

fn main() {}
