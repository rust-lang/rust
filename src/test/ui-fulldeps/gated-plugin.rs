// ignore-tidy-linelength
// aux-build:attr-plugin-test.rs

#![plugin(attr_plugin_test)]
//~^ ERROR compiler plugins are deprecated and will be removed in 1.44.0
//~| WARN use of deprecated attribute `plugin`: compiler plugins are deprecated and will be removed in 1.44.0

fn main() {}
