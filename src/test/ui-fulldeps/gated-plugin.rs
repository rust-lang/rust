// aux-build:attr-plugin-test.rs

#![plugin(attr_plugin_test)]
//~^ ERROR compiler plugins are experimental and possibly buggy

fn main() {}
