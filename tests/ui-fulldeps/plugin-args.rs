// aux-build:empty-plugin.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(empty_plugin(args))]
//~^ ERROR malformed `plugin` attribute
//~| WARNING compiler plugins are deprecated

fn main() {}
