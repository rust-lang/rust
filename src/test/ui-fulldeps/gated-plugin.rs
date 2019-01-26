// aux-build:basic_plugin.rs

#![plugin(basic_plugin)]
//~^ ERROR compiler plugins are experimental and possibly buggy

fn main() {}
