// aux-build:attr-plugin-test.rs

#![plugin(attr_plugin_test)]
//~^ ERROR compiler plugins are deprecated
//~| WARN use of deprecated attribute `plugin`: compiler plugins are deprecated

fn main() {}
